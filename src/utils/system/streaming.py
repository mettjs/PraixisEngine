import re
from collections.abc import AsyncGenerator, AsyncIterable

from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from src.utils.concurrency import GPUBusyError, SlotHandle
from src.utils.system.logger import logger


# Control markers are emitted by the streaming generators as standalone lines
# like ``[SESSION_ID:abc]\n``, ``[MODEL:fast]\n`` or ``[SOURCES:a,b]\n``,
# distinct from the LLM content tokens. A full match against this pattern is what separates a marker
# from a content token when draining a stream into a buffered JSON body.
# The keys are whitelisted: an LLM token that happens to look like
# ``[NOTE:x]\n`` must stay content, not be swallowed as a marker.
MARKER_KEYS = ("SESSION_ID", "MODEL", "SEARCH_QUERY", "SOURCES", "FILE", "PROGRESS", "ERROR")
_MARKER_RE = re.compile(r"\[(" + "|".join(MARKER_KEYS) + r"):(.*)\]\n?", re.DOTALL)

# Escaping for items inside the comma-separated ``[SOURCES:...]`` value.
# A filename may contain characters that corrupt either the list form (``,``)
# or the marker line itself (``]``, newlines); ``%`` is escaped first so
# decoding is unambiguous. Decoders split on ``,`` and then apply the reverse
# replacements in reverse order (``%25`` last). The SDK stream parsers
# implement the same table and must stay in sync.
_SOURCE_ESCAPES = (("%", "%25"), (",", "%2C"), ("]", "%5D"), ("\n", "%0A"), ("\r", "%0D"))


def encode_source_list(sources: list[str]) -> str:
    """Sources → the escaped, comma-joined ``[SOURCES:...]`` marker value."""
    encoded = []
    for source in sources:
        for raw, escaped in _SOURCE_ESCAPES:
            source = source.replace(raw, escaped)
        encoded.append(source)
    return ",".join(encoded)


def decode_source_list(value: str) -> list[str]:
    """The ``[SOURCES:...]`` marker value → the original source names."""
    sources = []
    for item in value.split(","):
        if not item:
            continue
        for raw, escaped in reversed(_SOURCE_ESCAPES):
            item = item.replace(escaped, raw)
        sources.append(item)
    return sources


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
            fields["sources"] = decode_source_list(value)
        elif key == "FILE":
            fields["filename"] = value
        else:
            fields[key.lower()] = value
    return {**fields, content_key: "".join(parts)}


async def guard_stream(content: AsyncIterable[str]) -> AsyncGenerator[str, None]:
    """Ends a live stream with an ``[ERROR:...]`` marker instead of a bare
    connection drop when the generator fails mid-flight.

    Headers are already sent by then, so an in-stream marker is the only way
    to tell the client "failed" apart from "done". Buffered paths must drain
    the raw generator instead, so exceptions map to real HTTP status codes.
    The GPUBusyError message is user-facing; anything else is logged and
    replaced with a generic marker.
    """
    try:
        async for piece in content:
            yield piece
    except GPUBusyError as e:
        yield f"[ERROR:{e}]\n"
    except Exception:
        logger.exception("Stream failed mid-flight.")
        yield "[ERROR:Internal error.]\n"


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
