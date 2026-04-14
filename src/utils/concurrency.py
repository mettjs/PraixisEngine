import os
import threading
from contextlib import contextmanager

# Maximum number of LLM calls that can run at the same time.
# Set this to 1 for a single-GPU setup running a large model.
# Increase it if your GPU has headroom for concurrent inference (e.g. small model, high VRAM).
_SLOTS = int(os.getenv("GPU_CONCURRENCY", "2"))
_GPU_SEMAPHORE = threading.BoundedSemaphore(_SLOTS)


class GPUBusyError(Exception):
    """Raised when all GPU slots are occupied and a new request cannot be served."""
    pass


@contextmanager
def gpu_slot():
    """
    Context manager for non-streaming LLM calls.

    Acquires a GPU slot immediately. Raises GPUBusyError (→ HTTP 503) if all
    slots are occupied so the caller fails fast instead of piling up threads.
    """
    acquired = _GPU_SEMAPHORE.acquire(blocking=False)
    if not acquired:
        raise GPUBusyError("All GPU slots are occupied. Please try again shortly.")
    try:
        yield
    finally:
        _GPU_SEMAPHORE.release()


def acquire_gpu_slot() -> None:
    """
    Eagerly acquires a GPU slot for a streaming response.

    Call this in the controller *before* constructing StreamingResponse so that
    a 503 can still be returned if the GPU is at capacity (once the response
    object is created and headers are sent, it is too late to change the status).

    Must be paired with release_gpu_slot() in a generator finally-block.
    """
    acquired = _GPU_SEMAPHORE.acquire(blocking=False)
    if not acquired:
        raise GPUBusyError("All GPU slots are occupied. Please try again shortly.")


def release_gpu_slot() -> None:
    """Releases a slot previously acquired with acquire_gpu_slot()."""
    _GPU_SEMAPHORE.release()
