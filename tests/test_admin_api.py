"""Admin endpoints (/api/system/*): basic auth, health, stats, API-key
lifecycle (generate → rotate → revoke), usage & daily buckets, session
wiping, audit log, and the vector-DB admin surface."""
import datetime
import uuid

from conftest import ADMIN_AUTH, VECTOR_BACKEND
from src.utils.store.api_keys import hash_api_key


def _generate_key(client, app_name: str) -> dict:
    response = client.post(
        "/api/system/keys/generate", params={"app_name": app_name}, auth=ADMIN_AUTH
    )
    assert response.status_code == 200
    return response.json()


# ── Auth & health ─────────────────────────────────────────────────────────────

def test_ping_needs_no_auth(client):
    assert client.get("/api/system/ping").json() == {"ok": True}


def test_admin_basic_auth(client):
    assert client.get("/api/system/auth/verify", auth=ADMIN_AUTH).json() == {"ok": True}
    assert client.get("/api/system/auth/verify").status_code == 401
    wrong = (ADMIN_AUTH[0], "wrong-password")
    assert client.get("/api/system/auth/verify", auth=wrong).status_code == 401


def test_health_reports_all_backends_online(client):
    body = client.get("/api/system/health", auth=ADMIN_AUTH).json()
    assert body == {"api": "online", "redis": "online", "vectordb": "online", "llm": "online"}
    assert client.get("/api/system/health/redis", auth=ADMIN_AUTH).json() == {"status": "online"}
    assert client.get("/api/system/health/vectordb", auth=ADMIN_AUTH).json() == {"status": "online"}


def test_stats_shape(client):
    body = client.get("/api/system/stats", auth=ADMIN_AUTH).json()
    assert isinstance(body["active_chat_sessions"], int)
    assert isinstance(body["total_vector_collections"], int)
    assert isinstance(body["total_vector_chunks"], int)


# ── API-key lifecycle ─────────────────────────────────────────────────────────

def test_key_generate_rotate_revoke_lifecycle(client, fake_llm):
    issued = _generate_key(client, "lifecycleapp")
    key = issued["api_key"]
    assert key.startswith("praixis_")
    key_hash = hash_api_key(key)

    listed = client.get("/api/system/keys", auth=ADMIN_AUTH).json()
    entry = next(e for e in listed["keys"] if e["key_hash"] == key_hash)
    assert entry["app_name"] == "lifecycleapp"
    assert entry["key_preview"] == key[:14] + "..."

    # The issued key authenticates as its app.
    response = client.get(
        "/general-requests/chat/sessions/active", headers={"X-API-Key": key}
    )
    assert response.status_code == 200

    # Rotation: the new key goes live, the old one is revoked.
    rotated = client.post(
        "/api/system/keys/rotate", params={"key_hash": key_hash}, auth=ADMIN_AUTH
    ).json()
    assert rotated["app_name"] == "lifecycleapp"
    assert rotated["old_key_revoked"] is True
    new_key = rotated["api_key"]
    assert new_key != key
    assert client.get(
        "/general-requests/chat/sessions/active", headers={"X-API-Key": new_key}
    ).status_code == 200
    assert client.get(
        "/general-requests/chat/sessions/active", headers={"X-API-Key": key}
    ).status_code == 403

    # Rotating an unknown hash must not mint a key.
    assert client.post(
        "/api/system/keys/rotate", params={"key_hash": "0" * 64}, auth=ADMIN_AUTH
    ).status_code == 404

    # Revocation kills the new key too.
    revoked = client.delete(
        "/api/system/keys/revoke-by-hash",
        params={"key_hash": hash_api_key(new_key)},
        auth=ADMIN_AUTH,
    )
    assert revoked.status_code == 200
    assert client.get(
        "/general-requests/chat/sessions/active", headers={"X-API-Key": new_key}
    ).status_code == 403
    assert client.delete(
        "/api/system/keys/revoke-by-hash",
        params={"key_hash": hash_api_key(new_key)},
        auth=ADMIN_AUTH,
    ).status_code == 404


# ── Usage ─────────────────────────────────────────────────────────────────────

def test_usage_totals_and_daily_buckets(client, headers, fake_llm):
    before = client.get("/api/system/usage/testapp", auth=ADMIN_AUTH).json()
    response = client.post(
        "/general-requests/chat",
        json={"prompt": "count me", "stream": False},
        headers=headers,
    )
    assert response.status_code == 200

    after = client.get("/api/system/usage/testapp", auth=ADMIN_AUTH).json()
    assert after["requests"] == before["requests"] + 1
    assert after["prompt_tokens"] == before["prompt_tokens"] + 7
    assert after["completion_tokens"] == before["completion_tokens"] + 5

    daily = client.get(
        "/api/system/usage/testapp/daily", params={"days": 2}, auth=ADMIN_AUTH
    ).json()
    assert daily["app_name"] == "testapp"
    assert len(daily["days"]) == 2
    today = daily["days"][0]
    assert today["date"] == datetime.datetime.now(datetime.UTC).date().isoformat()
    assert today["requests"] >= 1

    all_usage = client.get("/api/system/usage", auth=ADMIN_AUTH).json()
    assert any(app["app_name"] == "testapp" for app in all_usage["apps"])


# ── Sessions ──────────────────────────────────────────────────────────────────

def test_wipe_app_sessions(client, headers, redis_state, fake_llm):
    session_id = client.post(
        "/general-requests/chat", json={"prompt": "hi", "stream": False}, headers=headers
    ).json()["session_id"]
    assert redis_state.exists(f"chat:testapp:{session_id}")

    wiped = client.delete("/api/system/sessions/testapp", auth=ADMIN_AUTH).json()
    assert wiped["sessions_deleted"] >= 1
    assert not redis_state.exists(f"chat:testapp:{session_id}")
    assert redis_state.keys("usage:testapp:session:*") == []


# ── Audit log ─────────────────────────────────────────────────────────────────

def test_audit_records_key_and_file_events(client, headers, fake_llm):
    _generate_key(client, "auditapp")
    global_audit = client.get("/api/system/audit", auth=ADMIN_AUTH).json()
    assert any(
        event["action"] == "KEY_GENERATED" and event["details"]["app_name"] == "auditapp"
        for event in global_audit["events"]
    )

    collection = f"col-{uuid.uuid4().hex[:12]}"
    client.post(
        "/rag-db/upload_text",
        json={"text": "audit me", "filename": "audit.txt", "collection_name": collection,
              "chunking_strategy": "character"},
        headers=headers,
    )
    app_audit = client.get("/api/system/audit/testapp", auth=ADMIN_AUTH).json()
    assert any(
        event["action"] == "FILE_UPLOADED" and event["details"].get("filename") == "audit.txt"
        for event in app_audit["events"]
    )


# ── Vector-DB admin ───────────────────────────────────────────────────────────

def test_vector_admin_surface(client, headers):
    collection = f"col-{uuid.uuid4().hex[:12]}"
    client.post(
        "/rag-db/upload_text",
        json={"text": "Vacation policy: two days per month.", "filename": "admin.txt",
              "collection_name": collection, "chunking_strategy": "character"},
        headers=headers,
    )

    listing = client.get("/api/system/vector/collections", auth=ADMIN_AUTH).json()
    row = next(
        c for c in listing["collections"]
        if c["app_name"] == "testapp" and c["collection_name"] == collection
    )
    assert row["chunk_count"] >= 1

    files = client.get(
        f"/api/system/vector/collections/testapp/{collection}/files", auth=ADMIN_AUTH
    ).json()
    assert files["files"] == ["admin.txt"]

    search = client.get(
        "/api/system/vector/search",
        params={"app_name": "testapp", "collection_name": collection, "query": "vacation"},
        auth=ADMIN_AUTH,
    ).json()
    # Backend-dependent, same as the client-facing search endpoint: pgvector
    # reports RRF from its hybrid dense+FTS fusion, chroma raw similarity.
    assert search["score_type"] == ("rrf" if VECTOR_BACKEND == "pgvector" else "similarity")
    assert search["results"]

    deleted = client.delete(
        f"/api/system/vector/collections/testapp/{collection}/files",
        params={"filename": "admin.txt"},
        auth=ADMIN_AUTH,
    )
    assert deleted.status_code == 200
    assert client.delete(
        f"/api/system/vector/collections/testapp/{collection}/files",
        params={"filename": "admin.txt"},
        auth=ADMIN_AUTH,
    ).status_code == 404

    # Removing the last file removes the collection with it, on BOTH backends: a
    # collection exists exactly as long as it holds chunks. pgvector derives
    # collections from `chunks` rows; chroma drops the emptied collection to
    # match (see chroma/collections.py::delete_file_from_collection). So it is
    # already gone here, and is absent from listings too.
    listing_after = client.get("/api/system/vector/collections", auth=ADMIN_AUTH).json()
    assert not any(
        c["app_name"] == "testapp" and c["collection_name"] == collection
        for c in listing_after["collections"]
    )
    assert client.delete(
        f"/api/system/vector/collections/testapp/{collection}", auth=ADMIN_AUTH
    ).status_code == 404
