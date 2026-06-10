from openai import AsyncOpenAI
from src.config import AI_API_URL as _ai_api_url, AI_API_KEY as _ai_api_key
from src.utils.store.usage import record_usage
from src.utils.system.logger import logger


def get_async_ai_client() -> AsyncOpenAI:
    """Returns an async OpenAI-compatible client."""
    return AsyncOpenAI(base_url=_ai_api_url, api_key=_ai_api_key)


async def record_llm_usage(response, app_name: str) -> None:
    """Reads token counts from an OpenAI response and stores them in Redis."""
    try:
        usage = getattr(response, "usage", None)
        if usage:
            await record_usage(
                app_name=app_name,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
    except Exception as e:
        # Usage tracking must never break the main request, but a broken
        # counter should still be visible in the logs.
        logger.warning(f"Failed to record LLM usage for app {app_name}: {e}")
