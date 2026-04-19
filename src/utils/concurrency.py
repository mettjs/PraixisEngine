import asyncio
import os
from contextlib import asynccontextmanager

_SLOTS = int(os.getenv("GPU_CONCURRENCY", "2"))
_GPU_SEMAPHORE = asyncio.Semaphore(_SLOTS)


class GPUBusyError(Exception):
    """Raised when all GPU slots are occupied and a new request cannot be served."""
    pass


@asynccontextmanager
async def gpu_slot():
    """Async context manager for non-streaming LLM calls. Raises GPUBusyError immediately if full."""
    acquired = _GPU_SEMAPHORE._value > 0  # check without blocking
    if not acquired:
        raise GPUBusyError("All GPU slots are occupied. Please try again shortly.")
    async with _GPU_SEMAPHORE:
        yield


async def acquire_gpu_slot() -> None:
    """
    Eagerly acquires a GPU slot for a streaming response.

    Must be paired with release_gpu_slot() in a generator finally-block.
    Raises GPUBusyError if all slots are occupied.
    """
    if _GPU_SEMAPHORE._value <= 0:
        raise GPUBusyError("All GPU slots are occupied. Please try again shortly.")
    await _GPU_SEMAPHORE.acquire()


async def release_gpu_slot() -> None:
    """Releases a slot previously acquired with acquire_gpu_slot()."""
    _GPU_SEMAPHORE.release()
