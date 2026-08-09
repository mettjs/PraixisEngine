"""Admin endpoints (/api/system/*): basic auth, health, stats, API-key
lifecycle (generate → rotate → revoke), usage & daily buckets, session
wiping, audit log, and the vector-DB admin surface."""
import datetime
import uuid
from types import SimpleNamespace

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


def test_llm_health_pings_each_backend_once_not_each_model(client, pooled_models):
    """Three models, two endpoints: a proxy serving several models is one ping."""
    body = client.get("/api/system/health/llm", auth=ADMIN_AUTH).json()
    assert body["status"] == "online"
    backends = {entry["api_url"]: entry for entry in body["backends"]}
    assert len(backends) == 2
    assert backends["http://fake-llm.invalid"]["models"] == ["fast", "smart"]
    assert backends["http://fake-cloud.invalid"]["models"] == ["cloud"]


def test_llm_health_is_degraded_when_one_backend_is_down(client, pooled_models, monkeypatch):
    """One backend of several failing is not the same outage as having no LLM."""
    import src.controllers.admin_controller as admin_controller
    from conftest import FAKE_LLM

    class _Dead:
        """A backend whose ping never answers."""

        def __init__(self):
            self.models = SimpleNamespace(list=self._fail)

        def with_options(self, **_kwargs):
            return self

        async def _fail(self):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(
        admin_controller,
        "get_async_ai_client",
        lambda spec: _Dead() if spec.api_url == "http://fake-cloud.invalid" else FAKE_LLM,
    )
    body = client.get("/api/system/health/llm", auth=ADMIN_AUTH).json()
    assert body["status"] == "degraded"
    statuses = {entry["api_url"]: entry["status"] for entry in body["backends"]}
    assert statuses == {"http://fake-llm.invalid": "online", "http://fake-cloud.invalid": "offline"}
    # The rollup the dashboard reads reflects it too.
    assert client.get("/api/system/health", auth=ADMIN_AUTH).json()["llm"] == "degraded"


def test_model_registry_endpoint_describes_the_deployment(client, pooled_models):
    """The operator view: unscoped, with the backend and pool of each model —
    what the panel needs to render scoping and capacity."""
    body = client.get("/api/system/models", auth=ADMIN_AUTH).json()
    assert body["default"] == "fast"
    assert body["roles"] == {"utility": "fast", "background": "fast"}
    by_id = {m["id"]: m for m in body["models"]}
    assert by_id["smart"]["model"] == "qwen3:32b"
    assert by_id["smart"]["pool"] == "big"
    assert by_id["cloud"]["api_url"] == "http://fake-cloud.invalid"
    assert by_id["cloud"]["pool"] == "none"


def test_llm_health_pings_each_credential_on_a_shared_endpoint(client, monkeypatch):
    """Two models on one URL with different keys are two backends.

    Clients are cached on (api_url, api_key), so deduping the health check by
    URL alone would exercise one credential and report a revoked key as online.
    """
    import src.controllers.admin_controller as admin_controller
    import src.models.registry as registry_module
    from conftest import FAKE_LLM

    monkeypatch.setenv("KEY_A", "sk-a")
    monkeypatch.setenv("KEY_B", "sk-b")
    monkeypatch.setattr(registry_module, "_REGISTRY", registry_module.build_registry({
        "models": [
            {"id": "a", "model": "m", "api_url": "https://proxy.invalid/v1", "api_key": "${KEY_A}"},
            {"id": "b", "model": "n", "api_url": "https://proxy.invalid/v1", "api_key": "${KEY_B}"},
        ],
    }))

    class _Dead:
        def __init__(self):
            self.models = SimpleNamespace(list=self._fail)

        def with_options(self, **_kwargs):
            return self

        async def _fail(self):
            raise RuntimeError("401 unauthorized")

    # Only the revoked credential fails.
    monkeypatch.setattr(
        admin_controller,
        "get_async_ai_client",
        lambda spec: _Dead() if spec.api_key == "sk-a" else FAKE_LLM,
    )
    body = client.get("/api/system/health/llm", auth=ADMIN_AUTH).json()
    assert len(body["backends"]) == 2, body
    assert body["status"] == "degraded", body
    by_models = {entry["models"][0]: entry["status"] for entry in body["backends"]}
    assert by_models == {"a": "offline", "b": "online"}, body


def test_registry_can_be_written_and_read_back(client, tmp_path, monkeypatch):
    """The panel edits the file, not the running registry."""
    import src.models.registry as registry_module

    path = str(tmp_path / "models.yaml")
    monkeypatch.setattr(registry_module, "_MODELS_FILE", path)

    def _read(p):
        import os

        import yaml
        if not os.path.isfile(p):
            return None
        with open(p) as handle:
            return yaml.safe_load(handle)

    body = {"default": "fast", "models": [{"id": "fast", "model": "gemma4:e4b"}]}
    response = client.put("/api/system/models", json=body, auth=ADMIN_AUTH)
    assert response.status_code == 200, response.text
    assert response.json()["restart_required"] is True  # the file now differs from the live registry

    written = _read(path)
    assert written == body, written
    # Only what was typed is written — no env defaults baked in.
    assert "api_url" not in written["models"][0]


def test_registry_write_rejects_a_document_that_would_not_boot(client, tmp_path, monkeypatch):
    """Validation runs on the whole document before anything is written, so a
    save can never leave a registry the engine would refuse to start with."""
    import src.models.registry as registry_module

    path = str(tmp_path / "models.yaml")
    monkeypatch.setattr(registry_module, "_MODELS_FILE", path)

    good = {"models": [{"id": "fast", "model": "m"}]}
    assert client.put("/api/system/models", json=good, auth=ADMIN_AUTH).status_code == 200

    bad = {"default": "nope", "models": [{"id": "fast", "model": "m"}]}
    response = client.put("/api/system/models", json=bad, auth=ADMIN_AUTH)
    assert response.status_code == 400
    assert "default" in response.json()["detail"]

    import yaml
    with open(path) as handle:
        assert yaml.safe_load(handle) == good, "a rejected save must leave the file untouched"


def test_registry_write_falls_back_when_the_target_is_bind_mounted(client, tmp_path, monkeypatch):
    """Docker mounts models.yaml as its own mountpoint, so renaming over it is
    EBUSY regardless of ownership. The editor must still be able to save on the
    deployment shape the docs recommend."""
    import os

    import src.models.registry as registry_module

    path = str(tmp_path / "models.yaml")
    monkeypatch.setattr(registry_module, "_MODELS_FILE", path)
    assert client.put("/api/system/models", json={"models": [{"id": "a", "model": "m"}]},
                      auth=ADMIN_AUTH).status_code == 200

    def _ebusy(*_args, **_kwargs):
        raise OSError(16, "Device or resource busy")

    monkeypatch.setattr(os, "replace", _ebusy)
    response = client.put("/api/system/models", json={"models": [{"id": "b", "model": "n"}]},
                          auth=ADMIN_AUTH)
    assert response.status_code == 200, response.text

    import yaml
    with open(path) as handle:
        assert yaml.safe_load(handle) == {"models": [{"id": "b", "model": "n"}]}
    # And the fallback leaves no temp file behind.
    assert not [f for f in os.listdir(tmp_path) if f.startswith(".models.")]


def test_registry_can_be_removed(client, tmp_path, monkeypatch):
    """DELETE removes the file, returning the deployment to its env vars."""
    import os

    import src.models.registry as registry_module

    path = str(tmp_path / "models.yaml")
    monkeypatch.setattr(registry_module, "_MODELS_FILE", path)
    assert client.put("/api/system/models", json={"models": [{"id": "a", "model": "m"}]},
                      auth=ADMIN_AUTH).status_code == 200
    assert os.path.isfile(path)

    assert client.delete("/api/system/models", auth=ADMIN_AUTH).status_code == 200
    assert not os.path.isfile(path)


def test_registry_put_requires_a_body(client, tmp_path, monkeypatch):
    """Deleting the registry must never be reachable by forgetting a payload."""
    import os

    import src.models.registry as registry_module

    path = str(tmp_path / "models.yaml")
    monkeypatch.setattr(registry_module, "_MODELS_FILE", path)
    assert client.put("/api/system/models", json={"models": [{"id": "a", "model": "m"}]},
                      auth=ADMIN_AUTH).status_code == 200

    response = client.put("/api/system/models", auth=ADMIN_AUTH)
    assert response.status_code == 422, response.text
    assert os.path.isfile(path), "an empty PUT must not delete the registry"


def test_registry_endpoint_reports_the_file_and_restart_state(client, tmp_path, monkeypatch):
    import src.models.registry as registry_module

    path = str(tmp_path / "models.yaml")
    monkeypatch.setattr(registry_module, "_MODELS_FILE", path)

    # No file: the panel is told so, and nothing needs restarting.
    body = client.get("/api/system/models", auth=ADMIN_AUTH).json()
    assert body["file"] is None
    assert body["restart_required"] is False
    assert [m["id"] for m in body["models"]] == ["default"]

    client.put("/api/system/models", json={"models": [{"id": "fast", "model": "m"}]}, auth=ADMIN_AUTH)
    body = client.get("/api/system/models", auth=ADMIN_AUTH).json()
    assert body["file"] == {"models": [{"id": "fast", "model": "m"}]}
    # Still serving the old registry — that gap is the whole point of the flag.
    assert body["restart_required"] is True
    assert [m["id"] for m in body["models"]] == ["default"]


def test_registry_endpoint_reports_an_unreadable_file(client, tmp_path, monkeypatch):
    """A corrupt models.yaml must not be reported as an absent one — the panel
    would seed from env and overwrite something recoverable."""
    import src.models.registry as registry_module

    path = tmp_path / "models.yaml"
    path.write_text("models: [ unclosed\n")
    monkeypatch.setattr(registry_module, "_MODELS_FILE", str(path))

    body = client.get("/api/system/models", auth=ADMIN_AUTH).json()
    assert body["file"] is None
    assert body["file_error"] and "models.yaml" in body["file_error"]
    assert body["restart_required"] is True


def test_registry_endpoint_reports_writability(client, tmp_path, monkeypatch):
    """The panel says 'read-only' up front instead of discovering it on a 500."""

    import src.models.registry as registry_module

    path = tmp_path / "models.yaml"
    path.write_text("models:\n  - id: a\n    model: m\n")
    monkeypatch.setattr(registry_module, "_MODELS_FILE", str(path))
    assert client.get("/api/system/models", auth=ADMIN_AUTH).json()["writable"] is True

    path.chmod(0o444)
    try:
        assert client.get("/api/system/models", auth=ADMIN_AUTH).json()["writable"] is False
    finally:
        path.chmod(0o644)


def test_wiping_sessions_clears_bindings_with_no_history_left(client, headers, redis_state, fake_llm):
    """A history can expire while its model binding still has TTL left, so the
    wipe must not skip bindings just because no chat keys remain."""
    redis_state.set("session:orphanapp:deadbeef:model", "fast")
    assert redis_state.keys("chat:orphanapp:*") == []

    response = client.delete("/api/system/sessions/orphanapp", auth=ADMIN_AUTH)
    assert response.status_code == 200
    assert response.json()["sessions_deleted"] == 0
    assert redis_state.keys("session:orphanapp:*:model") == []


def test_gpu_reset_reports_every_pool(client, pooled_models):
    body = client.post("/api/system/gpu/reset", auth=ADMIN_AUTH).json()
    assert body["status"] == "success"
    assert body["slots_total"] == 2          # unchanged top-level contract
    assert body["pools"] == {
        "default": {"slots_total": 2, "hq_slots_total": 1},
        "big": {"slots_total": 1, "hq_slots_total": 0},
    }


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
