"""Global GPU concurrency control, enforced via Redis.

Two independent token buckets, each a Redis list:

* ``gpu:slots`` — the shared pool for interactive, user-facing LLM calls
  (chat, RAG answer, summary, compare), sized by ``GPU_CONCURRENCY``.
* ``gpu:hq_slots`` — a pool reserved exclusively for background hypothetical-
  question generation, sized by ``HQ_GPU_CONCURRENCY``.

Keeping them separate means a large document's question backfill can never drain
the slots that user requests depend on, and is itself never starved by live
traffic. Acquiring a slot pops a token (BLPOP, blocks up to the pool's timeout),
releasing pushes one back. Every worker and replica shares the same queues, so
each cap is a single global limit regardless of how many processes are running.
"""
import asyncio
from contextlib import asynccontextmanager

from src.config import (
    GPU_CONCURRENCY as _SLOTS,
    GPU_WAIT_TIMEOUT as _WAIT_TIMEOUT,
    HQ_GPU_CONCURRENCY as _HQ_SLOTS,
    HQ_GPU_WAIT_TIMEOUT as _HQ_WAIT_TIMEOUT,
)
from src.utils.store.client import gpu_redis_client as _redis

_GPU_QUEUE_KEY = "gpu:slots"
_GPU_INIT_KEY = "gpu:initialized"
_HQ_QUEUE_KEY = "gpu:hq_slots"
_HQ_INIT_KEY = "gpu:hq_initialized"

# (queue_key, init_key, slot_count) for every pool, so init/reset/status can
# iterate uniformly instead of duplicating logic per pool.
_POOLS = (
    (_GPU_QUEUE_KEY, _GPU_INIT_KEY, _SLOTS),
    (_HQ_QUEUE_KEY, _HQ_INIT_KEY, _HQ_SLOTS),
)


class GPUBusyError(Exception):
    """Raised when no slot frees up within the pool's wait timeout."""
    pass


async def init_gpu() -> None:
    """Populates each slot queue if it has not been sized for its current count.

    Called from the FastAPI lifespan hook on every process start. Each pool has a
    sentinel key storing the slot count it was last filled with; matching values
    skip the rebuild, so multi-worker and multi-replica deployments do not
    multiply the configured counts. A mismatch (e.g. a count changed in .env and
    the container was restarted) triggers a rebuild so config edits take effect
    without a manual ``/gpu/reset`` call.

    A consequence is that slots leaked by a hard crash persist across process
    restarts when the count is unchanged — recover them with
    ``POST /api/system/gpu/reset``.
    """
    for queue_key, init_key, slots in _POOLS:
        existing = await _redis.get(init_key)
        if existing != str(slots):
            await _fill_queue(queue_key, init_key, slots)


async def reset_gpu_counter() -> dict:
    """Forcibly rebuilds every pool to exactly its configured token count.

    Use after a crash leaks slots, or after changing a concurrency setting. Any
    in-flight request still holding an old token will push it back on release,
    transiently inflating a queue above its configured size until the next
    acquire drains the surplus.
    """
    for queue_key, init_key, slots in _POOLS:
        await _fill_queue(queue_key, init_key, slots)
    return {
        "status": "success",
        "message": "GPU slot counters reset.",
        "slots_total": _SLOTS,
        "hq_slots_total": _HQ_SLOTS,
    }


async def _fill_queue(queue_key: str, init_key: str, slots: int) -> None:
    pipe = _redis.pipeline()
    pipe.delete(queue_key)
    if slots > 0:
        pipe.rpush(queue_key, *(["1"] * slots))
    pipe.set(init_key, str(slots))
    await pipe.execute()


async def _acquire(queue_key: str, timeout: float) -> None:
    # BLPOP returns (key, value) when a token is popped, or None on timeout.
    result = await _redis.blpop([queue_key], timeout=timeout)
    if result is None:
        raise GPUBusyError("All GPU slots are occupied. Please try again shortly.")


async def _release(queue_key: str) -> None:
    await _redis.rpush(queue_key, "1")


@asynccontextmanager
async def gpu_slot():
    """Blocks until a shared (interactive) slot is free, then holds it for the duration."""
    await _acquire(_GPU_QUEUE_KEY, _WAIT_TIMEOUT)
    try:
        yield
    finally:
        await _release(_GPU_QUEUE_KEY)


@asynccontextmanager
async def hq_gpu_slot():
    """Holds a slot from the reserved question-generation pool.

    Waits up to ``HQ_GPU_WAIT_TIMEOUT`` (much longer than interactive, since this
    is best-effort background work). When ``HQ_GPU_CONCURRENCY`` is 0 the
    dedicated pool is disabled and generation falls back to the shared pool.
    """
    if _HQ_SLOTS <= 0:
        async with gpu_slot():
            yield
        return
    await _acquire(_HQ_QUEUE_KEY, _HQ_WAIT_TIMEOUT)
    try:
        yield
    finally:
        await _release(_HQ_QUEUE_KEY)


# Strong references to in-flight release tasks: a release started inside a
# cancelled scope (client disconnect) must survive its caller and complete.
_release_tasks: set[asyncio.Task] = set()


class SlotHandle:
    """A single acquired shared-pool permit.

    ``release()`` is idempotent — the streaming-response wrapper and any other
    owner can all call it, and exactly one token is pushed back. The push runs
    as a shielded task so cancellation of the caller (e.g. a client
    disconnecting mid-stream) cannot interrupt it and leak the permit.
    """

    __slots__ = ("_released",)

    def __init__(self) -> None:
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        task = asyncio.ensure_future(_release(_GPU_QUEUE_KEY))
        _release_tasks.add(task)
        task.add_done_callback(_release_tasks.discard)
        await asyncio.shield(task)


async def acquire_gpu_slot() -> SlotHandle:
    """Blocks until a shared slot is free (up to GPU_WAIT_TIMEOUT seconds).

    Used by streaming responses that must hold the slot across the entire
    stream. The returned handle's ``release()`` must be guaranteed to run —
    hand it to ``SlotReleasingStreamingResponse`` rather than relying on the
    body generator's ``finally`` (a generator closed before its first
    iteration never runs it).
    """
    await _acquire(_GPU_QUEUE_KEY, _WAIT_TIMEOUT)
    return SlotHandle()


async def get_gpu_status() -> dict:
    """Returns current slot usage for both pools, computed live from queue lengths."""
    available = int(await _redis.llen(_GPU_QUEUE_KEY))
    in_use = max(0, _SLOTS - available)
    hq_available = int(await _redis.llen(_HQ_QUEUE_KEY))
    hq_in_use = max(0, _HQ_SLOTS - hq_available)
    return {
        "slots_total": _SLOTS,
        "slots_in_use": in_use,
        "slots_available": available,
        "hq_slots_total": _HQ_SLOTS,
        "hq_slots_in_use": hq_in_use,
        "hq_slots_available": hq_available,
    }
