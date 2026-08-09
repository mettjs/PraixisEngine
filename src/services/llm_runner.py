"""Shared LLM execution primitives.

Every call here wraps the request in a GPU slot via ``gpu_slot()`` and records
token usage. Streaming endpoints that acquire the GPU slot in their controller
(chat, rag_answer) must NOT use these helpers — they would double-acquire.
"""
import asyncio
from collections.abc import AsyncGenerator

from src.config import CHUNK_CONCURRENCY as _CHUNK_CONCURRENCY
from src.models.registry import ModelSpec
from src.utils.ai_client import get_async_ai_client, record_llm_usage
from src.utils.concurrency import gpu_slot

_chunk_sem = asyncio.Semaphore(_CHUNK_CONCURRENCY)

Message = dict[str, str]


async def call_llm(
    messages: list[Message],
    app_name: str,
    spec: ModelSpec,
    gpu_ctx=gpu_slot,
    extra: dict | None = None,
    session_id: str | None = None,
) -> str:
    """Single non-streaming LLM call. Raises on empty response.

    ``spec`` is the registry entry to call — it carries the backend, the model
    name and any per-model ``params``, which a caller's ``extra`` overrides.
    ``gpu_ctx`` selects which bucket to draw a slot from — it defaults to the
    shared interactive one (``gpu_slot``). Background callers pass
    ``hq_gpu_slot`` so they consume the reserved question-generation bucket
    instead. Either way the slot comes from ``spec``'s pool, so a request never
    borrows capacity from hardware it is not using.
    ``extra`` is splatted into the request (e.g. ``response_format``).
    ``session_id`` attributes the token usage to that session as well.
    """
    async with gpu_ctx(spec.pool):
        response = await get_async_ai_client(spec).chat.completions.create(
            model=spec.model,
            messages=messages,  # type: ignore[arg-type]
            **{**spec.params, **(extra or {})},
        )
    await record_llm_usage(response, app_name, session_id=session_id, model_id=spec.id)
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("LLM returned no content.")
    return content


async def stream_llm(
    messages: list[Message],
    app_name: str,
    spec: ModelSpec,
    extra: dict | None = None,
    session_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Streams an LLM response token-by-token, holding the GPU slot for the
    full stream duration. ``spec`` selects the model. ``extra`` is splatted
    into the request (e.g. ``response_format``) and wins over the model's own
    ``params``. ``session_id`` attributes the token usage to that session as
    well."""
    async with gpu_slot(spec.pool):
        response = await get_async_ai_client(spec).chat.completions.create(
            model=spec.model,
            messages=messages,  # type: ignore[arg-type]
            stream=True,
            stream_options={"include_usage": True},
            **{**spec.params, **(extra or {})},
        )
        usage_recorded = False
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
            # Record once: a backend that attaches cumulative usage to more
            # than one chunk would otherwise be double-counted.
            if not usage_recorded and getattr(chunk, "usage", None):
                await record_llm_usage(chunk, app_name, session_id=session_id, model_id=spec.id)
                usage_recorded = True


async def map_calls_iter(
    messages_list: list[list[Message]], app_name: str, spec: ModelSpec
) -> AsyncGenerator[int | list[str], None]:
    """Runs many non-streaming calls concurrently, bounded by CHUNK_CONCURRENCY,
    yielding the running count of completed calls (1, 2, ...) as each finishes
    so callers can stream progress. The final yield is the ordered results list.
    On any failure — including the consuming generator being closed mid-flight —
    remaining tasks are cancelled so they release their GPU slots rather than
    running as orphaned background work."""
    async def _run(messages: list[Message]) -> str:
        async with _chunk_sem:
            return await call_llm(messages, app_name, spec)

    tasks = [asyncio.create_task(_run(m)) for m in messages_list]
    try:
        done = 0
        for finished in asyncio.as_completed(tasks):
            await finished
            done += 1
            yield done
        yield [task.result() for task in tasks]
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def map_calls(messages_list: list[list[Message]], app_name: str, spec: ModelSpec) -> list[str]:
    """Non-streaming wrapper over ``map_calls_iter`` for callers that don't
    report progress. Results preserve input order."""
    results: list[str] = []
    async for item in map_calls_iter(messages_list, app_name, spec):
        if isinstance(item, list):
            results = item
    return results
