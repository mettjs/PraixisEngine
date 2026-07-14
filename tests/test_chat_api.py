"""Chat & session endpoints (/general-requests/*) through the full app:
auth, buffered + streamed replies, session lifecycle, compaction, undo,
rate limiting, GPU-slot exhaustion, and file summaries."""
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
    assert response.text[match.end():] == DEFAULT_REPLY


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
