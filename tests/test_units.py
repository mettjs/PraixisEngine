"""Pure unit tests: streaming markers, SOURCES escaping, file parsing,
compaction sizing, the model registry, question parsing, API-key primitives.
No app, no HTTP."""
import asyncio
import hashlib
import io
import pathlib
import zipfile
from contextlib import asynccontextmanager

import pytest

from src.models.registry import (
    DEFAULT_POOL,
    ModelConfigError,
    UnknownModelError,
    build_registry,
    default_model,
    resolve_model,
    resolve_role,
)
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
    spec = default_model()
    small = [{"role": "user", "content": "x" * 1000}]
    huge = [{"role": "user", "content": "x" * 30000}]
    assert not needs_compaction(small, spec)
    assert needs_compaction(huge, spec)


def test_needs_compaction_is_measured_per_model():
    """The budget follows the model that will consume the history, so the same
    conversation compacts on a small model and not on a large one."""
    registry = build_registry({
        "models": [
            {"id": "small", "model": "m", "context_window": 4096},
            {"id": "large", "model": "m", "context_window": 128000},
        ]
    })
    history = [{"role": "user", "content": "x" * 20000}]  # ~5000 tokens
    assert needs_compaction(history, registry.get("small"))
    assert not needs_compaction(history, registry.get("large"))


# ── Model registry ────────────────────────────────────────────────────────────

def test_registry_without_a_file_synthesizes_one_model_from_env():
    """The zero-configuration path: no models.yaml, and the deployment still
    has exactly the model its env vars describe."""
    registry = build_registry(None)
    assert list(registry.models) == ["default"]
    spec = registry.default
    assert spec.model == "fake-model"        # MODEL_NAME in conftest
    assert spec.context_window == 8192       # CONTEXT_WINDOW in conftest
    assert spec.api_url == "http://fake-llm.invalid"
    assert spec.pool == DEFAULT_POOL
    # Both roles fall back to it, so internal chores work unconfigured.
    assert registry.roles == {"utility": "default", "background": "default"}


def test_registry_field_defaults_keep_entries_two_lines():
    registry = build_registry({"models": [{"id": "fast", "model": "gemma4:e4b"}]})
    spec = registry.get("fast")
    assert (spec.api_url, spec.api_key) == ("http://fake-llm.invalid", "test-key")
    assert spec.context_window == 8192
    assert spec.pool == DEFAULT_POOL
    assert spec.params == {}
    # A single-entry file needs no explicit 'default'.
    assert registry.default_id == "fast"


def test_registry_parses_a_full_file(monkeypatch):
    monkeypatch.setenv("REGISTRY_TEST_KEY", "sk-secret")
    registry = build_registry({
        "default": "smart",
        "roles": {"utility": "fast", "background": "fast"},
        "pools": {"big": 1},
        "models": [
            {"id": "fast", "model": "gemma4:e4b"},
            {"id": "smart", "model": "qwen3:32b", "context_window": 32768, "pool": "big"},
            {
                "id": "cloud",
                "model": "gpt-4o",
                "api_url": "https://api.openai.com/v1",
                "api_key": "${REGISTRY_TEST_KEY}",
                "pool": "none",
                "params": {"temperature": 0.2},
            },
        ],
    })
    assert registry.default_id == "smart"
    assert registry.roles["utility"] == "fast"
    assert registry.pools == {"big": 1}
    cloud = registry.get("cloud")
    assert cloud.api_key == "sk-secret"       # ${VAR} expanded
    assert cloud.params["temperature"] == 0.2
    assert not cloud.uses_gpu                 # remote: takes no local slot
    assert registry.get("smart").uses_gpu


def test_registry_rejects_unset_env_reference():
    """An unset ${VAR} must fail loudly instead of sending an empty API key."""
    with pytest.raises(ModelConfigError, match="REGISTRY_MISSING_KEY"):
        build_registry({"models": [
            {"id": "cloud", "model": "gpt-4o", "api_key": "${REGISTRY_MISSING_KEY}"}
        ]})


@pytest.mark.parametrize("raw, expected", [
    ({"models": []}, "non-empty list"),
    ({"models": [{"model": "m"}]}, "'id' is required"),
    ({"models": [{"id": "bad id", "model": "m"}]}, "must match"),
    ({"models": [{"id": "a", "model": "m"}, {"id": "a", "model": "n"}]}, "duplicate model id"),
    ({"default": "nope", "models": [{"id": "a", "model": "m"}]}, "'default' must name"),
    ({"roles": {"utility": "nope"}, "models": [{"id": "a", "model": "m"}]}, "undeclared model"),
    ({"roles": {"typo": "a"}, "models": [{"id": "a", "model": "m"}]}, "unknown role"),
    ({"models": [{"id": "a", "model": "m", "context_window": 0}]}, "positive integer"),
    ({"models": [{"id": "a", "model": "m", "temperature": 1}]}, "unknown field"),
    ({"pools": {"big": 0}, "models": [{"id": "a", "model": "m"}]}, "positive integer"),
    ("not a mapping", "top level must be a mapping"),
])
def test_registry_validation_failures_are_startup_errors(raw, expected):
    """Every malformed shape aborts at parse time — never as a runtime 500."""
    with pytest.raises(ModelConfigError, match=expected):
        build_registry(raw)


def test_resolve_model_falls_back_to_the_registry_default():
    assert resolve_model(None).id == default_model().id
    assert resolve_model("").id == default_model().id


def test_resolve_model_rejects_an_unknown_id_with_the_permitted_list():
    with pytest.raises(UnknownModelError) as exc:
        resolve_model("does-not-exist")
    assert exc.value.permitted == ["default"]
    assert "default" in str(exc.value)


def test_resolve_model_treats_a_forbidden_id_exactly_like_an_unknown_one():
    """A caller is never told a model exists that it may not use."""
    with pytest.raises(UnknownModelError) as exc:
        resolve_model("default", allowed=["other"])
    assert exc.value.permitted == []


def test_roles_resolve_independently_of_the_caller(monkeypatch):
    """Chores stay on the cheap model even when a request routes elsewhere."""
    import src.models.registry as registry_module

    registry = build_registry({
        "default": "smart",
        "roles": {"utility": "fast", "background": "fast"},
        "models": [{"id": "fast", "model": "m"}, {"id": "smart", "model": "n"}],
    })
    monkeypatch.setattr(registry_module, "_REGISTRY", registry)
    assert resolve_model(None).id == "smart"
    assert resolve_role("utility").id == "fast"
    assert resolve_role("background").id == "fast"


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


# ── GPU pool derivation ───────────────────────────────────────────────────────

def _pools_for(raw, monkeypatch):
    import src.models.registry as registry_module
    import src.utils.concurrency as concurrency

    monkeypatch.setattr(registry_module, "_REGISTRY", build_registry(raw))
    return concurrency._build_pools()


def test_default_deployment_keeps_exactly_one_pool(monkeypatch):
    """No models.yaml → the single 'default' pool with the original key names,
    so upgrading needs no Redis migration."""
    import src.utils.concurrency as concurrency

    slots, hq_slots = _pools_for(None, monkeypatch)
    assert slots == {"default": 2}          # GPU_CONCURRENCY in tests
    assert hq_slots == {"default": 1}       # HQ_GPU_CONCURRENCY
    assert concurrency._queue_key("default") == "gpu:slots"
    assert concurrency._hq_queue_key("default") == "gpu:hq_slots"
    assert concurrency._queue_key("big") == "gpu:slots:big"


def test_models_sharing_hardware_share_one_budget(monkeypatch):
    """Two models on one pool is one budget — the whole point of pool ids."""
    slots, _ = _pools_for({
        "pools": {"big": 1},
        "models": [
            {"id": "a", "model": "m", "pool": "big"},
            {"id": "b", "model": "n", "pool": "big"},
        ],
    }, monkeypatch)
    assert slots == {"big": 1, "default": 2}


def test_named_pool_without_a_declared_size_falls_back_to_gpu_concurrency(monkeypatch):
    slots, _ = _pools_for(
        {"models": [{"id": "a", "model": "m", "pool": "undeclared"}]}, monkeypatch
    )
    assert slots["undeclared"] == 2


def test_remote_models_get_no_bucket(monkeypatch):
    slots, hq_slots = _pools_for(
        {"models": [{"id": "cloud", "model": "gpt-4o", "pool": "none"}]}, monkeypatch
    )
    assert "none" not in slots and "none" not in hq_slots


def test_reserved_background_bucket_follows_the_background_role(monkeypatch):
    """HQ slots are reserved where background work actually runs; reserving
    them on a pool nothing schedules to would strand the capacity."""
    _, hq_slots = _pools_for({
        "default": "chat",
        "roles": {"background": "helper"},
        "pools": {"big": 1, "small": 4},
        "models": [
            {"id": "chat", "model": "m", "pool": "big"},
            {"id": "helper", "model": "n", "pool": "small"},
        ],
    }, monkeypatch)
    assert hq_slots == {"big": 0, "default": 0, "small": 1}


def test_declared_but_unused_pools_stay_visible(monkeypatch):
    """A pool no model references is kept, so a typo'd pool name shows up in
    /gpu instead of silently doing nothing."""
    slots, _ = _pools_for(
        {"pools": {"orphan": 3}, "models": [{"id": "a", "model": "m"}]}, monkeypatch
    )
    assert slots["orphan"] == 3


# ── Compaction and the utility pool ───────────────────────────────────────────

def test_compaction_takes_a_slot_on_the_utility_pool_when_it_differs(monkeypatch):
    """The caller's slot covers the chat model's backend, not the utility one.

    Without this the utility backend is called with no permit from its own
    pool, so N concurrent chats on another pool fire N unmetered requests at it.
    """
    import src.models.registry as registry_module
    import src.services.compaction as compaction

    registry = build_registry({
        "default": "chat",
        "roles": {"utility": "helper"},
        "pools": {"big": 1, "small": 4},
        "models": [
            {"id": "chat", "model": "m", "pool": "big"},
            {"id": "helper", "model": "n", "pool": "small"},
        ],
    })
    monkeypatch.setattr(registry_module, "_REGISTRY", registry)

    acquired: list[str] = []

    @asynccontextmanager
    async def _fake_slot(pool):
        acquired.append(pool)
        yield

    monkeypatch.setattr(compaction, "gpu_slot", _fake_slot)

    history = (
        [{"role": "system", "content": "be helpful"}]
        + [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}] * 5
    )
    asyncio.run(compaction.compact_history(history, app_name="app", session_id="s", held_pool="big"))
    assert acquired == ["small"], "must take a slot on the utility model's pool"


def test_compaction_reuses_the_held_slot_when_the_pool_matches(monkeypatch):
    """Same bucket: taking a second token while holding one deadlocks under
    load, so the held slot must be reused instead."""
    import src.models.registry as registry_module
    import src.services.compaction as compaction

    monkeypatch.setattr(registry_module, "_REGISTRY", build_registry(None))

    acquired: list[str] = []

    @asynccontextmanager
    async def _fake_slot(pool):
        acquired.append(pool)
        yield

    monkeypatch.setattr(compaction, "gpu_slot", _fake_slot)

    history = (
        [{"role": "system", "content": "be helpful"}]
        + [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}] * 5
    )
    asyncio.run(compaction.compact_history(history, app_name="app", session_id="s", held_pool="default"))
    assert acquired == [], "no second acquire on the pool the caller already holds"


# ── Registry file writes ──────────────────────────────────────────────────────

def _write(tmp_path, doc):
    from src.models.registry import write_registry_file
    path = str(tmp_path / "models.yaml")
    write_registry_file(doc, path)
    return path


def test_write_leaves_the_existing_file_alone_when_the_disk_is_full(tmp_path, monkeypatch):
    """The payload lands in a temp file first, so a failure there cannot touch
    a working registry — an emptied models.yaml reads as 'no registry' and would
    silently drop every declared model on the next restart."""
    import os

    from src.models.registry import load_registry, write_registry_file

    path = _write(tmp_path, {"models": [{"id": "a", "model": "m"}, {"id": "b", "model": "n"}]})
    before = pathlib.Path(path).read_text()

    def _no_space(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "fdopen", _no_space)
    with pytest.raises(OSError):
        write_registry_file({"models": [{"id": "a", "model": "m"}]}, path)
    monkeypatch.undo()

    assert pathlib.Path(path).read_text() == before
    assert list(load_registry(path).models) == ["a", "b"]
    assert not [f for f in os.listdir(tmp_path) if f.startswith(".models.")]


def test_write_falls_back_only_for_a_rename_that_cannot_work(tmp_path, monkeypatch):
    """EBUSY means the target is a mountpoint (Docker binds models.yaml as its
    own), which is the one case worth writing through. Anything else is a real
    failure and must not become an in-place write."""
    import os

    from src.models.registry import load_registry, write_registry_file

    path = _write(tmp_path, {"models": [{"id": "a", "model": "m"}]})

    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError(16, "busy")))
    write_registry_file({"models": [{"id": "b", "model": "n"}]}, path)
    assert list(load_registry(path).models) == ["b"]
    monkeypatch.undo()

    before = pathlib.Path(path).read_text()
    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError(13, "denied")))
    with pytest.raises(OSError):
        write_registry_file({"models": [{"id": "c", "model": "x"}]}, path)
    assert pathlib.Path(path).read_text() == before


def test_registry_file_state_separates_unreadable_from_absent(tmp_path):
    """An editor must never mistake a corrupt file for a missing one."""
    from src.models.registry import registry_file_state

    missing = str(tmp_path / "models.yaml")
    state = registry_file_state(missing)
    assert state["file"] is None and state["error"] is None
    assert state["matches_running"] is True and state["writable"] is True

    broken = tmp_path / "broken.yaml"
    broken.write_text("models: [ unclosed\n")
    state = registry_file_state(str(broken))
    assert state["file"] is None
    assert state["error"] and "broken.yaml" in state["error"]
    assert state["matches_running"] is False


def test_params_may_not_set_kwargs_the_engine_owns():
    """`params: {stream: false}` would be a TypeError on every request to that
    model; the error belongs at parse time, naming the file."""
    for reserved in ("stream", "model", "messages", "stream_options"):
        with pytest.raises(ModelConfigError, match="must not set"):
            build_registry({"models": [{"id": "a", "model": "m", "params": {reserved: 1}}]})
    # An ordinary param is still fine.
    spec = build_registry(
        {"models": [{"id": "a", "model": "m", "params": {"temperature": 0.2}}]}
    ).get("a")
    assert spec.params["temperature"] == 0.2


def test_pools_may_size_the_default_pool(monkeypatch):
    """models.yaml.example names the `default` pool, so sizing it there must
    apply rather than being silently discarded in favour of GPU_CONCURRENCY."""
    slots, _ = _pools_for({"pools": {"default": 8}, "models": [{"id": "a", "model": "m"}]}, monkeypatch)
    assert slots["default"] == 8
