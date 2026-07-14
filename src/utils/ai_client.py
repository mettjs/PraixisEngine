from openai import AsyncOpenAI
from src.config import AI_API_URL as _ai_api_url, AI_API_KEY as _ai_api_key
from src.utils.store.usage import record_usage
from src.utils.system.logger import logger


_client: AsyncOpenAI | None = None


def get_async_ai_client() -> AsyncOpenAI:
    """Returns the shared async OpenAI-compatible client.

    A process-wide singleton so every call site shares one httpx connection
    pool instead of each module holding its own. Callers needing different
    settings (e.g. the health check's short timeout) use ``with_options``,
    which shares the underlying pool.
    """
    global _client
    if _client is None:
        _client = AsyncOpenAI(base_url=_ai_api_url, api_key=_ai_api_key)
    return _client


async def record_llm_usage(response, app_name: str, session_id: str | None = None) -> None:
    """Reads token counts from an OpenAI response and stores them in Redis.

    When ``session_id`` is given, the tokens are additionally counted against
    that session (on top of the per-app totals)."""
    try:
        usage = getattr(response, "usage", None)
        if usage:
            await record_usage(
                app_name=app_name,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                session_id=session_id,
            )
    except Exception as e:
        # Usage tracking must never break the main request, but a broken
        # counter should still be visible in the logs.
        logger.warning(f"Failed to record LLM usage for app {app_name}: {e}")
