"""Chat & session endpoints (/general-requests/*) through the full app:
auth, buffered + streamed replies, session lifecycle, compaction, undo,
rate limiting, GPU-slot exhaustion, model selection, and file summaries."""
import re


from conftest import ADMIN_AUTH, DEFAULT_REPLY

_SESSION_ID_RE = re.compile(r"^\[SESSION_ID:([0-9a-f]{32})\]\n")


def _chat(client, headers, prompt="Hello there", session_id=None, stream=False, **extra):
    payload = {"prompt": prompt, "stream": stream, **extra}
    if session_id:
        payload["session_id"] = session_id
    return client.post("/general-requests/chat", json=payload, headers=headers)


# ── Authentication ────────────────────────────────────────────────────────────

def test_missing_api_key_is_rejected(client):
    response = client.post("/general-requests/chat", json={"prompt": "hi"})
    assert response.status_code == 403
    assert "missing" in response.json()["detail"].lower()


def test_invalid_api_key_is_rejected_and_counted(client, redis_state):
    response = client.post(
        "/general-requests/chat",
        json={"prompt": "hi"},
        headers={"X-API-Key": "praixis_forged-key"},
    )
    assert response.status_code == 403
    assert "invalid or revoked" in response.json()["detail"].lower()
    # The per-IP auth-fail counter must have been incremented.
    assert int(redis_state.get("authfail:testclient")) == 1


def test_auth_fail_flood_hits_429(client, redis_state):
    redis_state.set("authfail:testclient", 30)
    response = client.post(
        "/general-requests/chat",
        json={"prompt": "hi"},
        headers={"X-API-Key": "praixis_forged-key"},
    )
    assert response.status_code == 429


# ── Chat + session lifecycle ──────────────────────────────────────────────────

def test_buffered_chat_creates_session_with_history_and_usage(client, headers, fake_llm):
    response = _chat(client, headers, prompt="What is the leave policy?")
    assert response.status_code == 200
    body = response.json()
    session_id = body["session_id"]
    assert re.fullmatch(r"[0-9a-f]{32}", session_id)
    assert body["content"] == DEFAULT_REPLY

    history = client.get(f"/general-requests/chat/{session_id}", headers=headers).json()
    roles = [message["role"] for message in history["history"]]
    assert roles == ["system", "user", "assistant"]
    assert history["history"][1]["content"] == "What is the leave policy?"
    assert history["history"][2]["content"] == DEFAULT_REPLY

    usage = client.get(f"/general-requests/chat/{session_id}/usage", headers=headers).json()
    assert usage["requests"] == 1
    assert usage["prompt_tokens"] == 7
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 12
    assert usage["estimated_context_tokens"] > 0


def test_streaming_chat_emits_session_marker_then_tokens(client, headers, fake_llm):
    response = _chat(client, headers, stream=True)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    match = _SESSION_ID_RE.match(response.text)
    assert match, f"stream must start with a SESSION_ID marker, got: {response.text[:80]!r}"
    # SESSION_ID, then which model answered, then content.
    assert response.text[match.end():] == "[MODEL:default]\n" + DEFAULT_REPLY


def test_session_continuation_and_listing(client, headers, fake_llm):
    session_id = _chat(client, headers, prompt="first").json()["session_id"]
    second = _chat(client, headers, prompt="second", session_id=session_id).json()
    assert second["session_id"] == session_id

    history = client.get(f"/general-requests/chat/{session_id}", headers=headers).json()
    assert len(history["history"]) == 5  # system + 2 exchanges

    active = client.get("/general-requests/chat/sessions/active", headers=headers).json()
    assert session_id in active["active_sessions"]


def test_malformed_session_id_gets_a_fresh_session(client, headers, fake_llm):
    body = _chat(client, headers, session_id="not-a-real-session-id").json()
    assert re.fullmatch(r"[0-9a-f]{32}", body["session_id"])


def test_unknown_session_returns_404(client, headers):
    ghost = "deadbeef" * 4
    assert client.get(f"/general-requests/chat/{ghost}", headers=headers).status_code == 404
    assert client.get(f"/general-requests/chat/{ghost}/usage", headers=headers).status_code == 404
    assert client.post(f"/general-requests/chat/{ghost}/compact", headers=headers).status_code == 404
    assert client.delete(f"/general-requests/chat/{ghost}/last", headers=headers).status_code == 404


def test_undo_last_exchange(client, headers, fake_llm):
    session_id = _chat(client, headers, prompt="first question").json()["session_id"]
    _chat(client, headers, prompt="second question", session_id=session_id)

    undone = client.delete(f"/general-requests/chat/{session_id}/last", headers=headers).json()
    assert undone["removed_messages"] == 2
    assert undone["undone_prompt"] == "second question"
    assert undone["messages_remaining"] == 3

    client.delete(f"/general-requests/chat/{session_id}/last", headers=headers)
    # Only the system message is left — nothing more to undo.
    response = client.delete(f"/general-requests/chat/{session_id}/last", headers=headers)
    assert response.status_code == 400


def test_compact_requires_enough_history(client, headers, fake_llm):
    session_id = _chat(client, headers).json()["session_id"]
    response = client.post(f"/general-requests/chat/{session_id}/compact", headers=headers)
    assert response.status_code == 400


def test_compact_folds_older_exchanges_into_summary(client, headers, fake_llm):
    session_id = _chat(client, headers, prompt="exchange 1").json()["session_id"]
    for i in (2, 3, 4):
        _chat(client, headers, prompt=f"exchange {i}", session_id=session_id)

    fake_llm.queue("User asked about exchanges; nothing was resolved.")
    response = client.post(f"/general-requests/chat/{session_id}/compact", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["messages_before"] == 9   # system + 4 exchanges
    assert body["messages_after"] == 8    # system + summary + last 3 exchanges

    history = client.get(f"/general-requests/chat/{session_id}", headers=headers).json()["history"]
    assert history[1]["role"] == "system"
    assert history[1]["content"].startswith("[CONVERSATION SUMMARY]")
    assert "nothing was resolved" in history[1]["content"]


def test_clear_history_removes_session_and_usage(client, headers, redis_state, fake_llm):
    session_id = _chat(client, headers).json()["session_id"]
    assert redis_state.keys(f"usage:testapp:session:{session_id}:*")

    assert client.delete(f"/general-requests/chat/{session_id}", headers=headers).status_code == 200
    assert client.get(f"/general-requests/chat/{session_id}", headers=headers).status_code == 404
    assert client.delete(f"/general-requests/chat/{session_id}", headers=headers).status_code == 404
    assert redis_state.keys(f"usage:testapp:session:{session_id}:*") == []


# ── Failure modes ─────────────────────────────────────────────────────────────

def test_buffered_chat_maps_llm_failure_to_500(client, headers, fake_llm):
    fake_llm.queue(RuntimeError("LLM backend down"))
    assert _chat(client, headers).status_code == 500


def test_streaming_chat_surfaces_failure_as_error_marker(client, headers, fake_llm):
    # Headers are already sent when a live stream fails, so the failure must
    # arrive as an in-stream [ERROR:...] marker, not a broken connection.
    fake_llm.queue(RuntimeError("LLM backend down"))
    response = _chat(client, headers, stream=True)
    assert response.status_code == 200
    # The session marker is emitted before the LLM call, so it precedes the error.
    assert _SESSION_ID_RE.match(response.text)
    assert response.text.endswith("[ERROR:Internal error.]\n")


def test_chat_rate_limit_returns_429(client, fake_llm):
    # A dedicated key isolates this test's fixed-window bucket (10/minute).
    key = client.post(
        "/api/system/keys/generate", params={"app_name": "ratelimitapp"}, auth=ADMIN_AUTH
    ).json()["api_key"]
    limited_headers = {"X-API-Key": key}
    for _ in range(10):
        assert _chat(client, limited_headers).status_code == 200
    assert _chat(client, limited_headers).status_code == 429


def test_gpu_exhaustion_returns_503_until_reset(client, headers, redis_state, fake_llm):
    redis_state.delete("gpu:slots")  # simulate leaked slots
    assert _chat(client, headers).status_code == 503

    reset = client.post("/api/system/gpu/reset", auth=ADMIN_AUTH)
    assert reset.status_code == 200

    status = client.get("/api/system/gpu", auth=ADMIN_AUTH).json()
    assert status["slots_total"] == 2
    assert status["slots_available"] == 2
    assert _chat(client, headers).status_code == 200


# ── File summary ──────────────────────────────────────────────────────────────

def test_file_summary_buffered(client, headers, fake_llm):
    response = client.post(
        "/general-requests/file_summary",
        files={"file": ("notes.txt", b"Quarterly results were strong.", "text/plain")},
        data={"stream": "false"},
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "notes.txt"
    assert body["content"] == DEFAULT_REPLY


def test_file_summary_streaming_leads_with_file_marker(client, headers, fake_llm):
    response = client.post(
        "/general-requests/file_summary",
        files={"file": ("notes.txt", b"Quarterly results were strong.", "text/plain")},
        data={"stream": "true"},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.startswith("[FILE:notes.txt]\n")


def test_file_summary_rejects_empty_file(client, headers):
    response = client.post(
        "/general-requests/file_summary",
        files={"file": ("empty.txt", b"", "text/plain")},
        headers=headers,
    )
    assert response.status_code == 400


def test_file_summary_rejects_corrupted_pdf(client, headers):
    response = client.post(
        "/general-requests/file_summary",
        files={"file": ("broken.pdf", b"%PDF-1.4 not a real pdf", "application/pdf")},
        headers=headers,
    )
    assert response.status_code == 400
    assert "corrupted" in response.json()["detail"].lower()


def test_file_summary_rejects_oversized_file(client, headers):
    from src.config import MAX_FILE_SIZE

    response = client.post(
        "/general-requests/file_summary",
        files={"file": ("big.txt", b"x" * (MAX_FILE_SIZE + 1), "text/plain")},
        headers=headers,
    )
    assert response.status_code == 413


# ── Model selection ───────────────────────────────────────────────────────────

def test_models_endpoint_lists_the_synthesized_default(client, headers):
    """With no models.yaml the API still answers: one model, named 'default'."""
    response = client.get("/general-requests/models", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["default"] == "default"
    assert body["models"] == [{"id": "default", "context_window": 8192}]


def test_models_endpoint_reflects_the_registry(client, headers, multi_model):
    body = client.get("/general-requests/models", headers=headers).json()
    assert body["default"] == "smart"
    assert body["models"] == [
        {"id": "fast", "context_window": 8192},
        {"id": "smart", "context_window": 32768},
    ]


def test_chat_routes_to_the_requested_model(client, headers, fake_llm, multi_model):
    """The registry id selects which *backend* model name is actually called."""
    assert _chat(client, headers, model="fast").status_code == 200
    assert fake_llm.calls[-1]["model"] == "gemma4:e4b"

    fake_llm.reset()
    assert _chat(client, headers, model="smart").status_code == 200
    assert fake_llm.calls[-1]["model"] == "qwen3:32b"


def test_chat_without_a_model_uses_the_registry_default(client, headers, fake_llm, multi_model):
    assert _chat(client, headers).status_code == 200
    assert fake_llm.calls[-1]["model"] == "qwen3:32b"


def test_unknown_model_is_400_and_leaks_no_gpu_slot(client, headers, fake_llm, multi_model):
    response = _chat(client, headers, model="gpt-9")
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "gpt-9" in detail and "fast" in detail and "smart" in detail
    # Rejected before any LLM work, and without consuming a permit: the next
    # request still succeeds (GPU_CONCURRENCY is 2 in tests).
    assert not fake_llm.calls
    assert _chat(client, headers).status_code == 200


def test_model_id_charset_is_enforced_by_the_schema(client, headers):
    response = _chat(client, headers, model="not a valid id")
    assert response.status_code == 422


def test_file_summary_accepts_a_model_field(client, headers, fake_llm, multi_model):
    response = client.post(
        "/general-requests/file_summary",
        files={"file": ("notes.txt", b"Some short document text.", "text/plain")},
        data={"task": "Summarize", "model": "fast", "stream": "false"},
        headers=headers,
    )
    assert response.status_code == 200
    assert fake_llm.calls[-1]["model"] == "gemma4:e4b"


def test_compaction_runs_on_the_utility_model_not_the_chat_model(client, headers, fake_llm, multi_model):
    """Housekeeping stays cheap: a chat answered by 'smart' still has its
    summary written by the 'utility' role's model."""
    session_id = _chat(client, headers, prompt="first question", model="smart").json()["session_id"]
    for i in range(5):
        _chat(client, headers, prompt=f"question {i}", session_id=session_id, model="smart")

    fake_llm.reset()
    response = client.post(f"/general-requests/chat/{session_id}/compact", headers=headers)
    assert response.status_code == 200
    assert [call["model"] for call in fake_llm.calls] == ["gemma4:e4b"]


# ── GPU pools ─────────────────────────────────────────────────────────────────

def test_gpu_status_reports_every_pool(client, headers, pooled_models):
    status = client.get("/api/system/gpu", auth=ADMIN_AUTH).json()
    # The default pool's figures stay at the top level for existing consumers.
    assert status["slots_total"] == 2
    assert set(status["pools"]) == {"default", "big"}
    assert status["pools"]["big"]["slots_total"] == 1        # from the pools: block
    assert status["pools"]["default"]["hq_slots_total"] == 1  # background runs here
    assert status["pools"]["big"]["hq_slots_total"] == 0
    # A pool: none model owns no bucket at all.
    assert "none" not in status["pools"]


def test_pools_are_independent(client, headers, redis_state, fake_llm, pooled_models):
    """Draining one backend's budget must not stall a model on another."""
    redis_state.delete("gpu:slots:big")  # the 'big' pool is now exhausted
    assert _chat(client, headers, model="smart").status_code == 503
    assert _chat(client, headers, model="fast").status_code == 200


def test_remote_model_takes_no_local_slot(client, headers, redis_state, fake_llm, pooled_models):
    """pool: none runs on someone else's hardware: it must answer even with
    every local slot gone, and it must not leak a token by releasing one."""
    redis_state.delete("gpu:slots")
    assert _chat(client, headers, model="fast").status_code == 503

    assert _chat(client, headers, model="cloud").status_code == 200
    # Streaming takes the SlotReleasingStreamingResponse path, where a no-op
    # handle that pushed a token back would inflate the drained pool.
    streamed = _chat(client, headers, model="cloud", stream=True)
    assert streamed.status_code == 200
    assert streamed.text.endswith(DEFAULT_REPLY)
    assert redis_state.llen("gpu:slots") == 0


# ── Model marker & usage attribution ──────────────────────────────────────────

def test_streaming_chat_names_the_model_that_answered(client, headers, fake_llm, multi_model):
    response = _chat(client, headers, model="fast", stream=True)
    assert "[MODEL:fast]\n" in response.text
    # Markers lead the stream: content starts only after them.
    assert response.text.split("[MODEL:fast]\n")[1] == DEFAULT_REPLY


def test_buffered_chat_reports_the_model_as_a_field(client, headers, fake_llm, multi_model):
    body = _chat(client, headers, model="smart").json()
    assert body["model"] == "smart"
    assert body["content"] == DEFAULT_REPLY


def test_usage_is_attributed_per_model(client, headers, fake_llm, multi_model):
    def _by_model() -> dict:
        usage = client.get("/api/system/usage/testapp", auth=ADMIN_AUTH).json()
        # Counters are lifetime totals shared with the rest of the suite, so
        # this asserts on the delta the three calls below produce.
        return {row["model"]: row for row in usage["by_model"]}, usage

    before, _ = _by_model()
    _chat(client, headers, model="fast")
    _chat(client, headers, model="smart")
    _chat(client, headers, model="smart")
    after, usage = _by_model()

    def _delta(model_id: str, field: str) -> int:
        return after[model_id][field] - before.get(model_id, {}).get(field, 0)

    assert _delta("smart", "requests") == 2
    assert _delta("fast", "requests") == 1
    assert _delta("smart", "total_tokens") == 24  # 2 × (7 prompt + 5 completion)
    # The per-model rows are a split of the app totals, never larger.
    assert sum(row["requests"] for row in after.values()) <= usage["requests"]


# ── Per-key model scoping ─────────────────────────────────────────────────────

def _scoped_key(client, app_name: str, **params) -> str:
    response = client.post(
        "/api/system/keys/generate", params={"app_name": app_name, **params}, auth=ADMIN_AUTH
    )
    assert response.status_code == 200, response.text
    return response.json()["api_key"]


def test_scoped_key_sees_and_reaches_only_its_models(client, fake_llm, multi_model):
    key = _scoped_key(client, "scopedapp", models=["fast"], default_model="fast")
    scoped = {"X-API-Key": key}

    listing = client.get("/general-requests/models", headers=scoped).json()
    assert listing == {"models": [{"id": "fast", "context_window": 8192}], "default": "fast"}

    # A model outside the scope reads exactly like one that does not exist.
    denied = _chat(client, scoped, model="smart")
    assert denied.status_code == 400
    assert "smart" in denied.json()["detail"]
    assert "qwen3:32b" not in str(fake_llm.calls)

    # And the key's own default wins over the registry's ('smart').
    assert _chat(client, scoped).status_code == 200
    assert fake_llm.calls[-1]["model"] == "gemma4:e4b"


def test_unscoped_key_keeps_reaching_everything(client, fake_llm, multi_model):
    """Keys issued before scoping existed carry no allowlist and must not be
    narrowed by its arrival."""
    key = _scoped_key(client, "unscopedapp")
    unscoped = {"X-API-Key": key}
    assert len(client.get("/general-requests/models", headers=unscoped).json()["models"]) == 2
    assert _chat(client, unscoped, model="smart").status_code == 200


def test_generating_a_key_with_an_unknown_model_is_rejected(client, multi_model):
    """A typo surfaces to the admin issuing the key, not to the app later."""
    response = client.post(
        "/api/system/keys/generate",
        params={"app_name": "typoapp", "models": ["gpt-9"]},
        auth=ADMIN_AUTH,
    )
    assert response.status_code == 400
    assert "gpt-9" in response.json()["detail"]

    response = client.post(
        "/api/system/keys/generate",
        params={"app_name": "typoapp", "models": ["fast"], "default_model": "smart"},
        auth=ADMIN_AUTH,
    )
    assert response.status_code == 400
    assert "default_model" in response.json()["detail"]


def test_rotation_inherits_the_model_scope(client, fake_llm, multi_model):
    """Rotating must not silently widen what a key can reach."""
    _scoped_key(client, "rotatescope", models=["fast"], default_model="fast")
    key_hash = next(
        entry["key_hash"]
        for entry in client.get("/api/system/keys", auth=ADMIN_AUTH).json()["keys"]
        if entry["app_name"] == "rotatescope"
    )
    rotated = client.post("/api/system/keys/rotate", params={"key_hash": key_hash}, auth=ADMIN_AUTH).json()
    assert rotated["models"] == ["fast"]
    assert rotated["default_model"] == "fast"

    new_headers = {"X-API-Key": rotated["api_key"]}
    assert _chat(client, new_headers, model="smart").status_code == 400
    assert _chat(client, new_headers, model="fast").status_code == 200


def test_key_listing_exposes_the_scope(client, multi_model):
    _scoped_key(client, "listedscope", models=["fast", "smart"], default_model="smart")
    entry = next(
        e for e in client.get("/api/system/keys", auth=ADMIN_AUTH).json()["keys"]
        if e["app_name"] == "listedscope"
    )
    assert entry["models"] == ["fast", "smart"]
    assert entry["default_model"] == "smart"


# ── Session model binding ─────────────────────────────────────────────────────

def test_session_remembers_the_model_it_was_started_with(client, headers, fake_llm, multi_model):
    """A client escalates once; the rest of the conversation stays there."""
    session_id = _chat(client, headers, model="fast").json()["session_id"]
    assert fake_llm.calls[-1]["model"] == "gemma4:e4b"

    # No model on the follow-up: the binding wins over the registry default.
    body = _chat(client, headers, prompt="follow up", session_id=session_id).json()
    assert body["model"] == "fast"
    assert fake_llm.calls[-1]["model"] == "gemma4:e4b"


def test_explicit_model_rebinds_the_session(client, headers, fake_llm, multi_model):
    session_id = _chat(client, headers, model="fast").json()["session_id"]
    _chat(client, headers, prompt="escalate", session_id=session_id, model="smart")
    # Sticky afterwards: the override replaced the binding, not just this turn.
    body = _chat(client, headers, prompt="and again", session_id=session_id).json()
    assert body["model"] == "smart"


def test_binding_is_dropped_with_the_session(client, headers, redis_state, fake_llm, multi_model):
    session_id = _chat(client, headers, model="fast").json()["session_id"]
    assert redis_state.get(f"session:testapp:{session_id}:model") == "fast"
    client.delete(f"/general-requests/chat/{session_id}", headers=headers)
    assert redis_state.get(f"session:testapp:{session_id}:model") is None


def test_binding_a_key_may_no_longer_use_falls_back_instead_of_failing(
    client, headers, fake_llm, multi_model
):
    """Re-scoping a key should downgrade an old session, not break it — the
    caller did not ask for that model on this request."""
    session_id = _chat(client, headers, model="smart").json()["session_id"]

    # A second key for the SAME app, so the session (and its binding) is
    # visible to it, but scoped to a model the binding is not.
    scoped = {"X-API-Key": _scoped_key(client, "testapp", models=["fast"])}
    body = _chat(client, scoped, prompt="continue", session_id=session_id).json()
    assert body["session_id"] == session_id  # the binding really was in scope
    assert body["model"] == "fast"

    # ...and the fallback must not have rewritten the binding: the original,
    # unrestricted key still gets the model the session was started with.
    back = _chat(client, headers, prompt="and back", session_id=session_id).json()
    assert back["model"] == "smart"


def test_binding_ttl_survives_turns_that_do_not_rebind(client, headers, redis_state, fake_llm, multi_model):
    """A session that named its model once keeps it for the whole conversation.

    The history key is refreshed on every turn; if the binding were only
    written on the turn that named a model, it would expire underneath a long
    conversation and silently drop back to the default.
    """
    session_id = _chat(client, headers, model="fast").json()["session_id"]
    key = f"session:testapp:{session_id}:model"
    redis_state.expire(key, 5)  # simulate a binding close to expiry

    _chat(client, headers, prompt="follow up", session_id=session_id)
    assert redis_state.get(key) == "fast"
    assert redis_state.ttl(key) > 5, "a non-rebinding turn must still refresh the TTL"


def test_generating_a_key_with_a_blank_model_is_rejected(client, multi_model):
    """`?models=` yields [""] — an allowlist that can never resolve anything."""
    response = client.post(
        "/api/system/keys/generate",
        params={"app_name": "blankapp", "models": [""]},
        auth=ADMIN_AUTH,
    )
    assert response.status_code == 400
    assert "blank" in response.json()["detail"].lower()
