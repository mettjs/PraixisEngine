import re
from collections.abc import AsyncIterable

from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from src.utils.concurrency import SlotHandle


# Control markers are emitted by the streaming generators as standalone lines
# like ``[SESSION_ID:abc]\n`` or ``[SOURCES:a,b]\n``, distinct from the LLM
# content tokens. A full match against this pattern is what separates a marker
# from a content token when draining a stream into a buffered JSON body.
_MARKER_RE = re.compile(r"\[([A-Z_]+):(.*)\]\n?", re.DOTALL)


def parse_marker(piece: str) -> tuple[str, str] | None:
    """Returns ``(key, value)`` if ``piece`` is a control marker line, else
    ``None``. LLM content tokens do not match the full marker shape and so
    return ``None``."""
    match = _MARKER_RE.fullmatch(piece)
    if match is None:
        return None
    return match.group(1), match.group(2)


async def drain_to_json(content: AsyncIterable[str], content_key: str = "content") -> dict:
    """Consumes a streaming generator server-side and reassembles a JSON body.

    Control markers become structured fields (``SESSION_ID`` -> ``session_id``,
    ``SOURCES`` -> a list under ``sources``); content tokens are concatenated
    under ``content_key``. ``PROGRESS`` markers are dropped since they carry no
    meaning once the result is buffered. An in-stream ``[ERROR:...]`` marker is
    raised as a ``RuntimeError`` so the caller can return a real HTTP error
    instead of a 200 with an error string in the body.
    """
    fields: dict = {}
    parts: list[str] = []
    async for piece in content:
        parsed = parse_marker(piece)
        if parsed is None:
            parts.append(piece)
            continue
        key, value = parsed
        if key == "ERROR":
            raise RuntimeError(value)
        if key == "PROGRESS":
            continue
        if key == "SOURCES":
            fields["sources"] = [s for s in value.split(",") if s]
        elif key == "FILE":
            fields["filename"] = value
        else:
            fields[key.lower()] = value
    return {**fields, content_key: "".join(parts)}


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
