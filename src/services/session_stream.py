"""Shared session-aware streaming helpers for chat and RAG answers.

Both streaming endpoints follow the same shape: open (or resume) a session,
persist the user's message, stream the completion token-by-token, and persist
the assistant's (possibly partial) reply when the stream ends. The GPU slot is
owned by ``SlotReleasingStreamingResponse`` in the controller, not here.
"""
import asyncio
from collections.abc import AsyncGenerator

from src.models.registry import ModelSpec
from src.services.compaction import compact_history, needs_compaction
from src.utils.ai_client import get_async_ai_client, record_llm_usage
from src.utils.store.sessions import (
    bind_session_model,
    get_or_create_session,
    persist_history,
    touch_session_model,
)
from src.utils.system.logger import logger

# Strong references to in-flight persistence tasks: a persist started inside a
# cancelled scope (client disconnect) must survive its caller and complete.
_persist_tasks: set[asyncio.Task] = set()


async def open_user_turn(
    app_name: str,
    user_message: str,
    spec: ModelSpec,
    session_id: str | None = None,
    system_prompt: str | None = None,
    model_was_explicit: bool = False,
) -> tuple[str, list[dict[str, str]]]:
    """Resolves the session, appends the user message, and persists it.

    The session is also bound to ``spec``, but only when that is the caller's
    choice to make: a new session records what answered it, and an explicit
    per-turn override sticks for the turns that follow, so a client can escalate
    mid-conversation without repeating itself. A ``spec`` that is merely the
    *fallback* for a request naming no model must NOT overwrite an existing
    binding — a key scoped away from the bound model would otherwise rewrite the
    session for every other key that shares it. Its TTL is still refreshed, so
    the binding lives exactly as long as the history does.

    Persisting before the LLM call means the user's message survives even if
    generation fails afterwards. When the history approaches ``spec``'s context
    budget it is auto-compacted first — safe here because both callers (chat
    and RAG) already hold the GPU slot the compaction call needs. A compaction
    failure falls back to the uncompacted history rather than failing the turn.
    """
    active_session_id, history = await get_or_create_session(
        session_id=session_id,
        system_prompt=system_prompt,
        app_name=app_name,
    )
    # A returned id that differs from the requested one means the session was
    # just created (or the old one had expired), so there is no binding to
    # preserve and this turn's model is the one to record.
    is_new_session = active_session_id != (session_id or "")
    if model_was_explicit or is_new_session:
        await bind_session_model(app_name=app_name, session_id=active_session_id, model_id=spec.id)
    else:
        await touch_session_model(app_name=app_name, session_id=active_session_id)
    history.append({"role": "user", "content": user_message})
    if needs_compaction(history, spec):
        try:
            history = await compact_history(
                history, app_name=app_name, session_id=active_session_id, held_pool=spec.pool,
            )
            logger.info(f"Auto-compacted session {active_session_id} (app: {app_name}).")
        except Exception as e:
            logger.warning(
                f"Auto-compaction failed for session {active_session_id} (app: {app_name}): {e}"
            )
    await persist_history(app_name=app_name, session_id=active_session_id, history=history)
    return active_session_id, history


async def stream_assistant_turn(
    messages: list[dict[str, str]],
    app_name: str,
    session_id: str,
    history: list[dict[str, str]],
    spec: ModelSpec,
    extra: dict | None = None,
) -> AsyncGenerator[str, None]:
    """Streams a completion, then appends the accumulated reply to ``history``
    and persists it — including partial replies when the client disconnects.

    ``messages`` is what the LLM sees; ``history`` is what the session stores.
    RAG passes an augmented copy as ``messages`` so retrieved context is never
    persisted into the conversation. A persistence failure in the ``finally``
    is logged rather than raised so it cannot mask the stream's own outcome.
    """
    def _on_persist_done(task: asyncio.Task) -> None:
        _persist_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.warning(
                f"Failed to persist assistant reply for session {session_id} "
                f"(app: {app_name}): {task.exception()}"
            )

    full_response = ""
    try:
        response = await get_async_ai_client(spec).chat.completions.create(  # type: ignore[call-overload]
            model=spec.model,
            messages=messages,  # type: ignore[arg-type]
            stream=True,
            stream_options={"include_usage": True},
            **{**spec.params, **(extra or {})},
        )
        usage_recorded = False
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                token = chunk.choices[0].delta.content
                full_response += token
                yield token
            if not usage_recorded and getattr(chunk, "usage", None):
                await record_llm_usage(chunk, app_name, session_id=session_id, model_id=spec.id)
                usage_recorded = True
    finally:
        if full_response:
            history.append({"role": "assistant", "content": full_response})
            # A client disconnect cancels this generator's scope, and a bare
            # await here would be interrupted before the write completes. Run
            # the persist as a shielded task (same pattern as
            # ``SlotHandle.release``) so the partial reply is saved regardless;
            # failures are logged by the done callback, never raised.
            task = asyncio.ensure_future(
                persist_history(app_name=app_name, session_id=session_id, history=history)
            )
            _persist_tasks.add(task)
            task.add_done_callback(_on_persist_done)
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                raise  # the persist task itself continues in the background
            except Exception:
                pass  # already logged by _on_persist_done
