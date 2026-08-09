import uuid
import json
import re
from src.config import SESSION_TTL as _SESSION_TTL
from src.utils.store.client import redis_client
from src.utils.store.usage import delete_all_session_usage, delete_session_usage
from src.utils.system.logger import logger


def _get_redis_key(app_name: str, session_id: str) -> str:
    return f"chat:{app_name}:{session_id}"


def _model_key(app_name: str, session_id: str) -> str:
    """Sidecar key holding the model a session is bound to.

    A sidecar rather than a field inside the history: the history is a bare
    JSON list on the wire, and wrapping it in an envelope would mean migrating
    every live session. This keeps the format untouched and expires with it.
    """
    return f"session:{app_name}:{session_id}:model"


async def get_session_model(app_name: str, session_id: str) -> str | None:
    """The model this session is bound to, or None if it has never been set."""
    value = await redis_client.get(_model_key(app_name, session_id))
    return value if isinstance(value, str) else None


async def bind_session_model(app_name: str, session_id: str, model_id: str) -> None:
    """Binds (or rebinds) a session to a model, TTL-matched to the session."""
    await redis_client.setex(_model_key(app_name, session_id), _SESSION_TTL, model_id)


async def touch_session_model(app_name: str, session_id: str) -> None:
    """Refreshes an existing binding's TTL without changing what it points at.

    Every turn must do one of this or :func:`bind_session_model`, or a long
    conversation that named its model once would keep the history alive through
    ``persist_history`` while the binding quietly expired underneath it — and
    the session would fall back to the default mid-conversation.
    """
    await redis_client.expire(_model_key(app_name, session_id), _SESSION_TTL)


async def get_or_create_session(
    app_name: str,
    session_id: str | None = None,
    system_prompt: str | None = None,
) -> tuple[str, list[dict[str, str]]]:

    if not session_id or not re.fullmatch(r"[0-9a-f]{32}", session_id):
        session_id = None

    final_prompt = system_prompt or "You are a helpful institutional assistant."

    if session_id:
        redis_key = _get_redis_key(app_name, session_id)
        stored_data = await redis_client.get(redis_key)

        if isinstance(stored_data, str):
            history = json.loads(stored_data)

            if (system_prompt
                    and len(history) > 0
                    and history[0].get("role") == "system"
                    and history[0]["content"] != system_prompt):
                logger.warning(
                    f"Ignoring system_prompt override for existing session {session_id} "
                    f"(app: {app_name}). System prompt is fixed at session creation."
                )

            await redis_client.expire(redis_key, _SESSION_TTL)
            return session_id, history

    new_session_id = uuid.uuid4().hex
    new_redis_key = _get_redis_key(app_name, new_session_id)
    initial_history = [{"role": "system", "content": final_prompt}]
    await redis_client.setex(new_redis_key, _SESSION_TTL, json.dumps(initial_history))

    return new_session_id, initial_history


async def persist_history(app_name: str, session_id: str, history: list) -> None:
    """Writes an in-memory history back to Redis in a single round-trip.

    Use this when the caller already holds the history (e.g. from
    get_or_create_session) to avoid a redundant read-modify-write round-trip.
    History growth is bounded by compaction (see ``services.compaction``), not
    by trimming here, so nothing is dropped on write.
    """
    redis_key = _get_redis_key(app_name, session_id)
    await redis_client.setex(redis_key, _SESSION_TTL, json.dumps(history))


async def get_session_history(app_name: str, session_id: str) -> list:
    redis_key = _get_redis_key(app_name, session_id)
    data = await redis_client.get(redis_key)
    if isinstance(data, str):
        return json.loads(data)
    return []


async def delete_session(app_name: str, session_id: str) -> bool:
    redis_key = _get_redis_key(app_name, session_id)
    deleted = await redis_client.delete(redis_key) > 0  # type: ignore[operator]
    if deleted:
        await redis_client.delete(_model_key(app_name, session_id))
        await delete_session_usage(app_name=app_name, session_id=session_id)
    return deleted


async def get_all_active_sessions(app_name: str) -> list:
    prefix = f"chat:{app_name}:"
    prefix_length = len(prefix)
    keys = []
    async for key in redis_client.scan_iter(f"{prefix}*"):
        keys.append(str(key)[prefix_length:])
    return keys


async def delete_all_app_sessions(app_name: str) -> int:
    """Deletes all sessions for the given app (and their per-session usage
    counters). Returns the count of deleted session keys."""
    keys = [key async for key in redis_client.scan_iter(f"chat:{app_name}:*")]
    count = int(await redis_client.delete(*keys)) if keys else 0  # type: ignore[arg-type]
    # Not conditional on `keys`: a history can expire while its model binding
    # (written later in its own turn) still has TTL left, so an app with no
    # sessions can still have bindings to clear. Returning early here left them
    # behind for up to SESSION_TTL after a wipe reported success.
    bindings = [key async for key in redis_client.scan_iter(f"session:{app_name}:*:model")]
    if bindings:
        await redis_client.delete(*bindings)  # type: ignore[arg-type]
    await delete_all_session_usage(app_name)
    return count
