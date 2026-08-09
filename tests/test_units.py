"""Pure unit tests: streaming markers, SOURCES escaping, file parsing,
compaction sizing, question parsing, API-key primitives. No app, no HTTP."""
import asyncio
import hashlib
import io
import zipfile

import pytest

from src.services.compaction import compactable, estimate_tokens, needs_compaction
from src.utils.concurrency import GPUBusyError
from src.utils.file_parser import detect_file_kind, extract_text_from_file
from src.utils.store.api_keys import KEY_PREFIX, hash_api_key, new_api_key
from src.utils.system.streaming import (
    MARKER_KEYS,
    decode_source_list,
    drain_to_json,
    encode_source_list,
    guard_stream,
    parse_marker,
)
from src.utils.vectordb.questions import _parse_questions


# ── SOURCES escaping ──────────────────────────────────────────────────────────

def test_source_list_roundtrip():
    sources = [
        "plain.pdf",
        "Q3, Final.pdf",
        "closing]bracket.txt",
        "100% sure.md",
        "line\nbreak\rname.txt",
        "%,]",
    ]
    encoded = encode_source_list(sources)
    # No raw delimiter may survive inside an item.
    assert "]" not in encoded and "\n" not in encoded and "\r" not in encoded
    assert "Q3%2C Final.pdf" in encoded
    assert decode_source_list(encoded) == sources


def test_source_decode_applies_percent_last():
    # A filename that literally contains an escape sequence must round-trip:
    # "%2C" is encoded as "%252C" and must NOT decode to ",".
    assert encode_source_list(["%2C"]) == "%252C"
    assert decode_source_list("%252C") == ["%2C"]


def test_source_list_skips_empty_items():
    assert decode_source_list("") == []
    assert decode_source_list("a,,b") == ["a", "b"]


# ── Marker parsing ────────────────────────────────────────────────────────────

def test_parse_marker_whitelists_keys():
    for key in MARKER_KEYS:
        assert parse_marker(f"[{key}:value]\n") == (key, "value")
    # An LLM token that merely looks like a marker must stay content.
    assert parse_marker("[NOTE:x]\n") is None
    assert parse_marker("plain token") is None
    assert parse_marker("[SESSION_ID:abc] trailing") is None


def test_drain_to_json_collects_markers_and_content():
    async def _gen():
        yield "[SESSION_ID:abc123]\n"
        yield "[SEARCH_QUERY:vacation days]\n"
        yield f"[SOURCES:{encode_source_list(['Q3, Final.pdf'])}]\n"
        yield "[PROGRESS:mapping 2 chunks]\n"
        yield "Hello "
        yield "world."

    body = asyncio.run(drain_to_json(_gen()))
    assert body == {
        "session_id": "abc123",
        "search_query": "vacation days",
        "sources": ["Q3, Final.pdf"],
        "content": "Hello world.",
    }


def test_drain_to_json_raises_on_error_marker():
    async def _gen():
        yield "partial "
        yield "[ERROR:boom]\n"

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(drain_to_json(_gen()))


def test_guard_stream_converts_failures_to_error_markers():
    async def _explodes():
        yield "tok "
        raise ValueError("secret internals")

    async def _gpu_busy():
        raise GPUBusyError("All GPU slots are occupied.")
        yield  # pragma: no cover — makes this a generator

    async def _collect(gen):
        return [piece async for piece in gen]

    pieces = asyncio.run(_collect(guard_stream(_explodes())))
    # Internal details must not leak into the stream.
    assert pieces == ["tok ", "[ERROR:Internal error.]\n"]

    pieces = asyncio.run(_collect(guard_stream(_gpu_busy())))
    assert pieces == ["[ERROR:All GPU slots are occupied.]\n"]


# ── File parsing ──────────────────────────────────────────────────────────────

def test_detect_file_kind_precedence():
    # Extension wins over a contradictory content type.
    assert detect_file_kind("a.pdf", b"hello", content_type="text/plain") == "pdf"
    # Content type is the fallback for extension-less names.
    assert detect_file_kind("noext", b"hello", content_type="text/plain; charset=utf-8") == "txt"
    # application/octet-stream carries no information.
    assert detect_file_kind("noext", b"hello", content_type="application/octet-stream") is None
    # Magic bytes are the last resort.
    assert detect_file_kind("noext", b"%PDF-1.7 ...") == "pdf"
    assert detect_file_kind("noext", b"PK\x03\x04rest") == "docx"
    assert detect_file_kind("noext", b"\x00\x01\x02") is None


def test_corrupted_pdf_raises_clean_value_error():
    with pytest.raises(ValueError, match="Unsupported or corrupted"):
        extract_text_from_file("broken.pdf", b"%PDF-1.4 this is not a real pdf")


def test_non_docx_zip_raises_clean_value_error():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.csv", "a,b,c")
    with pytest.raises(ValueError, match="Unsupported or corrupted"):
        extract_text_from_file("archive", buffer.getvalue())


def test_txt_falls_back_to_latin1():
    assert extract_text_from_file("x.txt", "café".encode("latin-1")) == "café"


def test_unknown_format_rejected():
    with pytest.raises(ValueError, match="Unsupported file format"):
        extract_text_from_file("data.bin", b"\x00\x01\x02\x03")


# ── Compaction sizing ─────────────────────────────────────────────────────────

def test_estimate_tokens_is_chars_over_four():
    history = [{"role": "user", "content": "x" * 400}, {"role": "assistant", "content": "y" * 100}]
    assert estimate_tokens(history) == 125


def test_needs_compaction_threshold():
    # CONTEXT_WINDOW=8192 in tests → trigger at 6553 tokens (~26 214 chars).
    small = [{"role": "user", "content": "x" * 1000}]
    huge = [{"role": "user", "content": "x" * 30000}]
    assert not needs_compaction(small)
    assert needs_compaction(huge)


def test_compactable_requires_history_beyond_recent_window():
    system = [{"role": "system", "content": "be helpful"}]
    pair = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert not compactable(system + pair * 3)  # all within the verbatim window
    assert compactable(system + pair * 5)


# ── Hypothetical-question parsing ─────────────────────────────────────────────

def test_parse_questions_strips_markers_dedups_and_limits():
    raw = (
        "- How many vacation days do I get?\n"
        "1. How many vacation days do I get?\n"
        "• short\n"
        "(2) Who approves my leave request?\n"
        "* When can I take my first day off?\n"
    )
    assert _parse_questions(raw, limit=5) == [
        "How many vacation days do I get?",
        "Who approves my leave request?",
        "When can I take my first day off?",
    ]
    assert _parse_questions(raw, limit=2) == [
        "How many vacation days do I get?",
        "Who approves my leave request?",
    ]


def test_parse_questions_keeps_leading_digits_of_real_questions():
    assert _parse_questions("2024 rules: what changed for me?", limit=5) == [
        "2024 rules: what changed for me?"
    ]


# ── API-key primitives ────────────────────────────────────────────────────────

def test_new_api_key_prefix_and_uniqueness():
    first, second = new_api_key(), new_api_key()
    assert first.startswith(KEY_PREFIX) and second.startswith(KEY_PREFIX)
    assert first != second


def test_hash_api_key_is_sha256_hex():
    key = "praixis_example"
    assert hash_api_key(key) == hashlib.sha256(key.encode()).hexdigest()


# ── Golden marker vectors ─────────────────────────────────────────────────────
# tests/marker_vectors.json is the shared contract with the three SDKs; see the
# _comment block inside it. These tests hold the ENGINE side: the whitelist and
# the [SOURCES:...] escape table. Each SDK vendors the same file and asserts the
# same behaviour against its own decoder, so a change made here without copying
# the file outward fails an ordinary test over there.

def test_marker_vectors_whitelist(marker_vectors):
    """Every whitelisted key parses as a marker; nothing else does."""
    for key in marker_vectors["marker_keys"]:
        assert parse_marker(f"[{key}:value]\n") == (key, "value"), key

    for key in marker_vectors["non_marker_keys"]:
        assert parse_marker(f"[{key}:value]\n") is None, (
            f"'{key}' must stay LLM content — widening the whitelist silently "
            f"swallows tokens that merely look like markers"
        )


def test_marker_vectors_whitelist_matches_implementation(marker_vectors):
    """The vectors enumerate exactly the keys the engine emits.

    Adding a marker means updating this file and re-vendoring it to all three
    SDKs; this assertion is what makes forgetting that impossible to miss.
    """
    assert list(MARKER_KEYS) == marker_vectors["marker_keys"]


def test_marker_vectors_source_escaping(marker_vectors):
    """Encode and decode both match the contract, for every vector."""
    for case in marker_vectors["source_escaping"]:
        decoded, encoded = case["decoded"], case["encoded"]
        assert encode_source_list(decoded) == encoded, case["name"]
        assert decode_source_list(encoded) == decoded, case["name"]


def test_marker_vectors_source_escaping_round_trips(marker_vectors):
    """decode(encode(x)) == x — the property the table exists to guarantee."""
    for case in marker_vectors["source_escaping"]:
        decoded = case["decoded"]
        assert decode_source_list(encode_source_list(decoded)) == decoded, case["name"]
