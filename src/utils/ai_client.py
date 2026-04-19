import os
from openai import OpenAI, AsyncOpenAI

_ai_api_url = os.getenv("AI_API_URL", "http://localhost:8081")
_ai_api_key = os.getenv("AI_API_KEY", "")


def get_ai_client() -> OpenAI:
    """Returns a sync OpenAI-compatible client (used for health checks)."""
    return OpenAI(base_url=_ai_api_url, api_key=_ai_api_key)


def get_async_ai_client() -> AsyncOpenAI:
    """Returns an async OpenAI-compatible client."""
    return AsyncOpenAI(base_url=_ai_api_url, api_key=_ai_api_key)


async def record_llm_usage(response, app_name: str) -> None:
    """Reads token counts from an OpenAI response and stores them in Redis."""
    try:
        from src.utils.memory import record_usage  # local import avoids circular dependency
        usage = getattr(response, "usage", None)
        if usage:
            await record_usage(
                app_name=app_name,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
    except Exception:
        pass  # usage tracking must never break the main request
