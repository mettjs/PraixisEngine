"""Global GPU concurrency control, enforced via Redis.

Capacity is tracked per **pool**, where a pool is one physical backend. Models
served by the same hardware share a pool id and therefore share its budget;
a model marked ``pool: none`` (a remote/cloud endpoint) consumes no local
capacity at all and takes no token.

Each pool owns two independent token buckets, each a Redis list:

* ``gpu:slots[:<pool>]`` — the shared pool for interactive, user-facing LLM
  calls (chat, RAG answer, summary, compare), sized by ``GPU_CONCURRENCY`` or,
  for a named pool, by the ``pools:`` block in ``models.yaml``.
* ``gpu:hq_slots[:<pool>]`` — reserved exclusively for background hypothetical-
  question generation, sized by ``HQ_GPU_CONCURRENCY``. Only the pool that the
  ``background`` role's model belongs to gets one; elsewhere background work
  falls back to the shared bucket, exactly as it does when
  ``HQ_GPU_CONCURRENCY`` is 0.

Keeping the two separate means a large document's question backfill can never
drain the slots that user requests depend on, and is itself never starved by
live traffic. Acquiring a slot pops a token (BLPOP, blocks up to the pool's
timeout), releasing pushes one back. Every worker and replica shares the same
queues, so each cap is a single global limit regardless of how many processes
are running.

The ``default`` pool deliberately keeps the unsuffixed key names it has always
used, so upgrading a single-model deployment needs no Redis migration.
"""
import asyncio
from contextlib import asynccontextmanager

from src.config import (
    GPU_CONCURRENCY as _SLOTS,
    GPU_WAIT_TIMEOUT as _WAIT_TIMEOUT,
    HQ_GPU_CONCURRENCY as _HQ_SLOTS,
    HQ_GPU_WAIT_TIMEOUT as _HQ_WAIT_TIMEOUT,
)
from src.models.registry import DEFAULT_POOL, NO_POOL, list_models, pool_sizes, resolve_role
from src.utils.store.client import gpu_redis_client as _redis

_GPU_QUEUE_KEY = "gpu:slots"
_GPU_INIT_KEY = "gpu:initialized"
_HQ_QUEUE_KEY = "gpu:hq_slots"
_HQ_INIT_KEY = "gpu:hq_initialized"


def _build_pools() -> tuple[dict[str, int], dict[str, int]]:
    """Derives every pool's budget from the registry.

    A pool exists if a model draws from it or the ``pools:`` block declares it
    — declared-but-unused pools stay visible in ``/gpu`` rather than silently
    doing nothing, which is how a typo'd pool name gets noticed. Any pool with
    no declared size falls back to ``GPU_CONCURRENCY``, including ``default``:
    naming ``default`` in ``pools:`` overrides the env var for it rather than
    being quietly discarded, since models.yaml.example names that pool and an
    operator sizing it there gets no other signal.
    """
    declared = dict(pool_sizes())
    names = {DEFAULT_POOL, *declared}
    names.update(spec.pool for spec in list_models() if spec.uses_gpu)
    slots = {name: declared.get(name, _SLOTS) for name in sorted(names)}
    # The reserved background bucket belongs to whichever pool actually runs
    # background work; giving every pool one would reserve capacity nothing
    # can ever use.
    background_pool = resolve_role("background").pool
    hq_slots = {name: (_HQ_SLOTS if name == background_pool else 0) for name in slots}
    return slots, hq_slots


_POOL_SLOTS, _POOL_HQ_SLOTS = _build_pools()


def _queue_key(pool: str) -> str:
    return _GPU_QUEUE_KEY if pool == DEFAULT_POOL else f"{_GPU_QUEUE_KEY}:{pool}"


def _init_key(pool: str) -> str:
    return _GPU_INIT_KEY if pool == DEFAULT_POOL else f"{_GPU_INIT_KEY}:{pool}"


def _hq_queue_key(pool: str) -> str:
    return _HQ_QUEUE_KEY if pool == DEFAULT_POOL else f"{_HQ_QUEUE_KEY}:{pool}"


def _hq_init_key(pool: str) -> str:
    return _HQ_INIT_KEY if pool == DEFAULT_POOL else f"{_HQ_INIT_KEY}:{pool}"


def _buckets() -> list[tuple[str, str, int]]:
    """(queue_key, init_key, slot_count) for every bucket of every pool, so
    init/reset can iterate uniformly."""
    buckets = []
    for pool, slots in _POOL_SLOTS.items():
        buckets.append((_queue_key(pool), _init_key(pool), slots))
        buckets.append((_hq_queue_key(pool), _hq_init_key(pool), _POOL_HQ_SLOTS[pool]))
    return buckets


class GPUBusyError(Exception):
    """Raised when no slot frees up within the pool's wait timeout."""
    pass


async def init_gpu() -> None:
    """Populates each slot queue if it has not been sized for its current count.

    Called from the FastAPI lifespan hook on every process start. Each bucket
    has a sentinel key storing the slot count it was last filled with; matching
    values skip the rebuild, so multi-worker and multi-replica deployments do
    not multiply the configured counts. A mismatch (e.g. a count changed in
    .env, or a pool resized in models.yaml, and the container was restarted)
    triggers a rebuild so config edits take effect without a manual
    ``/gpu/reset`` call.

    A consequence is that slots leaked by a hard crash persist across process
    restarts when the count is unchanged — recover them with
    ``POST /api/system/gpu/reset``.
    """
    for queue_key, init_key, slots in _buckets():
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
    for queue_key, init_key, slots in _buckets():
        await _fill_queue(queue_key, init_key, slots)
    return {
        "status": "success",
        "message": "GPU slot counters reset.",
        # Top-level figures stay the default pool's, so existing consumers
        # (the admin panel among them) keep reading what they always read.
        "slots_total": _POOL_SLOTS[DEFAULT_POOL],
        "hq_slots_total": _POOL_HQ_SLOTS[DEFAULT_POOL],
        "pools": {
            pool: {"slots_total": slots, "hq_slots_total": _POOL_HQ_SLOTS[pool]}
            for pool, slots in _POOL_SLOTS.items()
        },
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
async def gpu_slot(pool: str = DEFAULT_POOL):
    """Blocks until a shared (interactive) slot in ``pool`` is free, then holds
    it for the duration.

    ``pool: none`` models run on someone else's hardware, so there is nothing
    to serialize: the block runs immediately and unmetered.
    """
    if pool == NO_POOL:
        yield
        return
    queue_key = _queue_key(pool)
    await _acquire(queue_key, _WAIT_TIMEOUT)
    try:
        yield
    finally:
        await _release(queue_key)


@asynccontextmanager
async def hq_gpu_slot(pool: str = DEFAULT_POOL):
    """Holds a slot from ``pool``'s reserved question-generation bucket.

    Waits up to ``HQ_GPU_WAIT_TIMEOUT`` (much longer than interactive, since
    this is best-effort background work). When the pool has no reserved bucket
    — ``HQ_GPU_CONCURRENCY`` is 0, or background work runs on a different pool
    than this one — generation falls back to that pool's shared bucket.
    """
    if pool == NO_POOL:
        yield
        return
    if _POOL_HQ_SLOTS.get(pool, 0) <= 0:
        async with gpu_slot(pool):
            yield
        return
    queue_key = _hq_queue_key(pool)
    await _acquire(queue_key, _HQ_WAIT_TIMEOUT)
    try:
        yield
    finally:
        await _release(queue_key)


# Strong references to in-flight release tasks: a release started inside a
# cancelled scope (client disconnect) must survive its caller and complete.
_release_tasks: set[asyncio.Task] = set()


class SlotHandle:
    """A single acquired shared-pool permit.

    ``release()`` is idempotent — the streaming-response wrapper and any other
    owner can all call it, and exactly one token is pushed back. The push runs
    as a shielded task so cancellation of the caller (e.g. a client
    disconnecting mid-stream) cannot interrupt it and leak the permit.

    A handle with ``queue_key=None`` is the no-op form handed out for
    ``pool: none`` models: it satisfies the same interface, so streaming a
    remote model goes through exactly the same code path as a local one.
    """

    __slots__ = ("_released", "_queue_key")

    def __init__(self, queue_key: str | None = _GPU_QUEUE_KEY) -> None:
        self._released = False
        self._queue_key = queue_key

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._queue_key is None:
            return
        task = asyncio.ensure_future(_release(self._queue_key))
        _release_tasks.add(task)
        task.add_done_callback(_release_tasks.discard)
        await asyncio.shield(task)


async def acquire_gpu_slot(pool: str = DEFAULT_POOL) -> SlotHandle:
    """Blocks until a shared slot in ``pool`` is free (up to GPU_WAIT_TIMEOUT).

    Used by streaming responses that must hold the slot across the entire
    stream. The returned handle's ``release()`` must be guaranteed to run —
    hand it to ``SlotReleasingStreamingResponse`` rather than relying on the
    body generator's ``finally`` (a generator closed before its first
    iteration never runs it). For a ``pool: none`` model the handle is a no-op
    and nothing is acquired.
    """
    if pool == NO_POOL:
        return SlotHandle(queue_key=None)
    queue_key = _queue_key(pool)
    await _acquire(queue_key, _WAIT_TIMEOUT)
    return SlotHandle(queue_key)


async def get_gpu_status() -> dict:
    """Current slot usage for every pool, computed live from queue lengths."""
    pools = list(_POOL_SLOTS)
    pipe = _redis.pipeline()
    for pool in pools:
        pipe.llen(_queue_key(pool))
        pipe.llen(_hq_queue_key(pool))
    lengths = await pipe.execute()

    per_pool: dict[str, dict] = {}
    for index, pool in enumerate(pools):
        available, hq_available = int(lengths[2 * index]), int(lengths[2 * index + 1])
        total, hq_total = _POOL_SLOTS[pool], _POOL_HQ_SLOTS[pool]
        per_pool[pool] = {
            "slots_total": total,
            "slots_in_use": max(0, total - available),
            "slots_available": available,
            "hq_slots_total": hq_total,
            "hq_slots_in_use": max(0, hq_total - hq_available),
            "hq_slots_available": hq_available,
        }
    # The default pool's figures stay at the top level: every existing consumer
    # reads them there, and on a single-model deployment they are the whole story.
    return {**per_pool[DEFAULT_POOL], "pools": per_pool}
