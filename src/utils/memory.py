import uuid
import os
import json
import re
import redis.asyncio as aioredis
from typing import List, Dict, Tuple

_REDIS = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = aioredis.Redis.from_url(_REDIS, decode_responses=True)

# Session expiry in seconds
_SESSION_TTL: int = int(os.getenv("SESSION_TTL", 86400))
# Max user+assistant message pairs to keep before trimming. System prompt is always preserved.
_MAX_HISTORY_PAIRS: int = int(os.getenv("MAX_HISTORY_PAIRS", 20))


def _get_redis_key(app_name: str, session_id: str) -> str:
    return f"chat:{app_name}:{session_id}"


def _trim_history(history: list) -> list:
    """Keeps the system prompt and the most recent MAX_HISTORY_PAIRS exchange pairs."""
    system = [m for m in history if m["role"] == "system"]
    messages = [m for m in history if m["role"] != "system"]
    max_messages = _MAX_HISTORY_PAIRS * 2
    if len(messages) > max_messages:
        messages = messages[-max_messages:]
    return system + messages


async def get_or_create_session(
    app_name: str,
    session_id: str | None = None,
    system_prompt: str | None = None,
) -> Tuple[str, List[Dict[str, str]]]:

    if not session_id or not re.fullmatch(r"[0-9a-f]{32}", session_id):
        session_id = None

    final_prompt = system_prompt or "You are a helpful institutional assistant."

    if session_id:
        redis_key = _get_redis_key(app_name, session_id)
        stored_data = await redis_client.get(redis_key)

        if isinstance(stored_data, str):
            history = json.loads(stored_data)

            if system_prompt and len(history) > 0 and history[0].get("role") == "system":
                history[0]["content"] = system_prompt
                await redis_client.setex(redis_key, _SESSION_TTL, json.dumps(history))
            else:
                await redis_client.expire(redis_key, _SESSION_TTL)

            return session_id, history

    new_session_id = uuid.uuid4().hex
    new_redis_key = _get_redis_key(app_name, new_session_id)
    initial_history = [{"role": "system", "content": final_prompt}]
    await redis_client.setex(new_redis_key, _SESSION_TTL, json.dumps(initial_history))

    return new_session_id, initial_history


async def add_message(app_name: str, session_id: str, role: str, content: str):
    redis_key = _get_redis_key(app_name, session_id)
    stored_data = await redis_client.get(redis_key)

    if isinstance(stored_data, str):
        history = json.loads(stored_data)
        history.append({"role": role, "content": content})
        history = _trim_history(history)
        await redis_client.setex(redis_key, _SESSION_TTL, json.dumps(history))


async def get_session_history(app_name: str, session_id: str) -> list:
    redis_key = _get_redis_key(app_name, session_id)
    data = await redis_client.get(redis_key)
    if isinstance(data, str):
        return json.loads(data)
    return []


async def delete_session(app_name: str, session_id: str) -> bool:
    redis_key = _get_redis_key(app_name, session_id)
    return await redis_client.delete(redis_key) > 0  # type: ignore[operator]


async def get_all_active_sessions(app_name: str) -> list:
    prefix = f"chat:{app_name}:"
    prefix_length = len(prefix)
    keys = []
    async for key in redis_client.scan_iter(f"{prefix}*"):
        keys.append(str(key)[prefix_length:])
    return keys


async def record_usage(app_name: str, prompt_tokens: int, completion_tokens: int) -> None:
    pipe = redis_client.pipeline()
    pipe.incrby(f"usage:{app_name}:prompt_tokens", prompt_tokens)
    pipe.incrby(f"usage:{app_name}:completion_tokens", completion_tokens)
    pipe.incrby(f"usage:{app_name}:requests", 1)
    await pipe.execute()


async def get_usage(app_name: str) -> dict:
    pipe = redis_client.pipeline()
    pipe.get(f"usage:{app_name}:prompt_tokens")
    pipe.get(f"usage:{app_name}:completion_tokens")
    pipe.get(f"usage:{app_name}:requests")
    prompt, completion, requests = await pipe.execute()
    return {
        "app_name": app_name,
        "requests": int(requests or 0),
        "prompt_tokens": int(prompt or 0),
        "completion_tokens": int(completion or 0),
        "total_tokens": int(prompt or 0) + int(completion or 0),
    }


async def get_all_app_names() -> list[str]:
    """Returns every app_name that has a usage record."""
    app_names: set[str] = set()
    async for key in redis_client.scan_iter("usage:*:requests"):
        parts = str(key).split(":")
        if len(parts) >= 2:
            app_names.add(parts[1])
    return list(app_names)


async def delete_all_app_sessions(app_name: str) -> int:
    """Deletes all sessions for the given app. Returns the count of deleted keys."""
    keys = [key async for key in redis_client.scan_iter(f"chat:{app_name}:*")]
    if not keys:
        return 0
    return int(await redis_client.delete(*keys))  # type: ignore[arg-type]
