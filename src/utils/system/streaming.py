from collections.abc import AsyncIterable

from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from src.utils.concurrency import SlotHandle


class SlotReleasingStreamingResponse(StreamingResponse):
    """StreamingResponse that owns a GPU slot and guarantees its release.

    An async generator that is closed before its first iteration never runs
    its ``finally`` block, so release logic inside the body generator alone
    leaks the permit when a client disconnects before streaming starts.
    Releasing here, after the ASGI call finishes, covers every path: normal
    completion, mid-stream errors, and immediate disconnects. ``release()``
    is idempotent, so an extra call from another owner is harmless.
    """

    def __init__(self, content: AsyncIterable[str], slot: SlotHandle, **kwargs) -> None:
        super().__init__(content, **kwargs)
        self._slot = slot

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._slot.release()
