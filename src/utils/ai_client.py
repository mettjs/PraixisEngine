from openai import AsyncOpenAI

from src.models.registry import ModelSpec
from src.utils.store.usage import record_usage
from src.utils.system.logger import logger


# One client per *endpoint*, not per model: several registry entries usually
# share a single LiteLLM/vLLM proxy, and they should share its connection pool.
_clients: dict[tuple[str, str], AsyncOpenAI] = {}


def _new_client(api_url: str, api_key: str) -> AsyncOpenAI:
    """Client construction, isolated so tests can substitute a fake backend
    without having to predict which endpoints the registry will ask for."""
    return AsyncOpenAI(base_url=api_url, api_key=api_key)


def get_async_ai_client(spec: ModelSpec) -> AsyncOpenAI:
    """Returns the shared async OpenAI-compatible client for ``spec``'s backend.

    Cached on ``(api_url, api_key)``, so every call site targeting the same
    endpoint shares one httpx connection pool instead of each module holding
    its own. Callers needing different settings (e.g. the health check's short
    timeout) use ``with_options``, which shares the underlying pool.
    """
    key = (spec.api_url, spec.api_key)
    client = _clients.get(key)
    if client is None:
        client = _new_client(*key)
        _clients[key] = client
    return client


async def record_llm_usage(
    response, app_name: str, session_id: str | None = None, model_id: str | None = None
) -> None:
    """Reads token counts from an OpenAI response and stores them in Redis.

    When ``session_id`` is given, the tokens are additionally counted against
    that session (on top of the per-app totals); ``model_id`` adds the same
    tokens to that model's lifetime totals, which is what makes cost
    attribution across a multi-model registry possible."""
    try:
        usage = getattr(response, "usage", None)
        if usage:
            await record_usage(
                app_name=app_name,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                session_id=session_id,
                model_id=model_id,
            )
    except Exception as e:
        # Usage tracking must never break the main request, but a broken
        # counter should still be visible in the logs.
        logger.warning(f"Failed to record LLM usage for app {app_name}: {e}")
