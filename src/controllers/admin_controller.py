import asyncio
from fastapi import HTTPException
from src.utils.store.client import redis_client
from src.utils.store.sessions import delete_all_app_sessions
from src.utils.store.usage import get_usage, get_all_app_names, get_daily_usage, get_usage_by_model
from src.utils.store.api_keys import (
    store_api_key,
    remove_api_key_by_hash,
    list_all_api_keys,
    get_api_key_entry,
    new_api_key,
)
from src.utils.vectordb import get_vector_store
from src.models.registry import (
    ROLES,
    ModelConfigError,
    ModelSpec,
    UnknownModelError,
    default_model,
    list_models,
    registry_file_state,
    resolve_model,
    resolve_role,
    write_registry_file,
)
from src.utils.ai_client import get_async_ai_client
from src.utils.concurrency import get_gpu_status, reset_gpu_counter
from src.utils.store.audit import log_event, get_audit_log
from src.utils.system.logger import logger

def _llm_backends() -> list[tuple[ModelSpec, list[str]]]:
    """(one spec to ping, the model ids it stands for) per distinct backend.

    Deduped by endpoint *and credential*, matching how ai_client caches its
    clients: a dozen models behind one LiteLLM proxy on one key is a single
    ping, but two models sharing a URL with different ``api_key``s are two —
    pinging only one of them would report a revoked key as healthy.
    """
    backends: dict[tuple[str, str], tuple[ModelSpec, list[str]]] = {}
    for spec in list_models():
        _, ids = backends.setdefault((spec.api_url, spec.api_key), (spec, []))
        ids.append(spec.id)
    return list(backends.values())


async def get_redis_health() -> dict:
    try:
        await redis_client.ping()  # type: ignore[misc]
        return {"status": "online"}
    except Exception:
        logger.error("Redis health check failed.")
        return {"status": "offline"}


async def get_vectordb_health() -> dict:
    try:
        await get_vector_store().ping()
        return {"status": "online"}
    except Exception:
        logger.error("Vector DB health check failed.")
        return {"status": "offline"}


async def get_llm_health() -> dict:
    """Pings every configured LLM backend once.

    ``status`` is ``online`` when they all answer, ``offline`` when none do,
    and ``degraded`` in between — a registry where one of three backends is
    down is not the same outage as having no LLM at all, and the difference
    matters to whoever is paged.
    """
    async def _ping(spec) -> bool:
        try:
            await get_async_ai_client(spec).with_options(timeout=5.0).models.list()
            return True
        except Exception:
            return False

    backends = _llm_backends()
    results = await asyncio.gather(*[_ping(spec) for spec, _ in backends])

    detail = [
        {"api_url": spec.api_url, "models": sorted(model_ids), "status": "online" if ok else "offline"}
        for (spec, model_ids), ok in zip(backends, results)
    ]
    online = sum(1 for ok in results if ok)
    if online == len(results):
        status = "online"
    elif online == 0:
        status = "offline"
        logger.error("Every LLM backend failed its health check.")
    else:
        status = "degraded"
        offline = [d["api_url"] for d in detail if d["status"] == "offline"]
        logger.error(f"LLM backend health check failed for: {', '.join(offline)}")
    return {"status": status, "backends": detail}


async def get_health_status() -> dict:
    redis_s, vectordb_s, llm_s = await asyncio.gather(
        get_redis_health(), get_vectordb_health(), get_llm_health()
    )
    return {"api": "online", "redis": redis_s["status"], "vectordb": vectordb_s["status"], "llm": llm_s["status"]}


async def get_system_stats() -> dict:
    async def _count_sessions():
        # Sessions expire via TTL, so a maintained counter would drift; a
        # cursor SCAN stays exact. The large COUNT hint keeps it to ~one
        # round-trip per 1000 keys instead of Redis's default of 10.
        count = 0
        async for _ in redis_client.scan_iter("chat:*", count=1000):
            count += 1
        return count

    active_sessions, (num_collections, total_vectors) = await asyncio.gather(
        _count_sessions(),
        get_vector_store().stats(),
    )
    return {
        "active_chat_sessions": active_sessions,
        "total_vector_collections": num_collections,
        "total_vector_chunks": total_vectors,
    }


def _validate_model_scope(models: list[str] | None, default_model_id: str | None) -> None:
    """Rejects a scope that names models this deployment does not have.

    Checked at issue time so a typo surfaces here, to an admin who can fix it,
    rather than as a 400 on the app's first request.
    """
    for model_id in models or []:
        # resolve_model treats a falsy id as "unspecified" and hands back the
        # default, so a blank entry would sail through and store an allowlist
        # of [""] — a key that can never resolve anything and 400s forever.
        if not model_id.strip():
            raise HTTPException(status_code=400, detail="Model ids must not be blank.")
        try:
            resolve_model(model_id)
        except UnknownModelError as e:
            raise HTTPException(status_code=400, detail=str(e))
    if default_model_id:
        try:
            resolve_model(default_model_id, allowed=models or None)
        except UnknownModelError:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"default_model '{default_model_id}' must be a configured model"
                    + (" listed in models." if models else ".")
                ),
            )


async def generate_api_key(
    app_name: str, models: list[str] | None = None, default_model: str | None = None
) -> dict:
    # Named default_model to match the query parameter; bound to a local that
    # does not shadow the registry accessor imported above.
    default_model_id = default_model
    _validate_model_scope(models, default_model_id)
    full_key = new_api_key()
    await store_api_key(full_key, app_name, models=models, default_model=default_model_id)
    await log_event("KEY_GENERATED", {"app_name": app_name, "models": models or "all"})
    logger.info(f"Generated new API Key for app: {app_name}")
    return {
        "app_name": app_name,
        "api_key": full_key,
        "models": models or [],
        "default_model": default_model_id,
        "message": "Store this key safely. It will not be shown again.",
    }


async def list_api_keys() -> dict:
    entries = await list_all_api_keys()
    return {"total_keys": len(entries), "keys": entries}


async def delete_app_sessions(app_name: str) -> dict:
    """Force-expires all Redis sessions belonging to a specific app."""
    count = await delete_all_app_sessions(app_name)
    await log_event("SESSION_WIPED", {"sessions_deleted": count}, app_name=app_name)
    logger.info(f"Wiped {count} session(s) for app: {app_name}")
    return {"status": "success", "sessions_deleted": count, "app_name": app_name}


async def rotate_api_key(key_hash: str) -> dict:
    """Replaces a key in one call: the new key goes live before the old one is
    revoked, so the app never has a window with zero valid keys."""
    entry = await get_api_key_entry(key_hash)
    app_name = (entry or {}).get("app_name")
    if not app_name:
        raise HTTPException(status_code=404, detail="API Key not found.")
    full_key = new_api_key()
    # The replacement inherits the old key's model scope: rotating a key must
    # not silently widen what it can reach.
    await store_api_key(
        full_key,
        app_name,
        models=entry.get("models"),
        default_model=entry.get("default_model"),
    )
    # From here the new key is live. A failure must not become a 500 that never
    # delivers it — the caller would retry and mint yet another key while the
    # stored one sits orphaned. Report the revocation outcome instead.
    old_key_revoked = False
    try:
        await remove_api_key_by_hash(key_hash)
        old_key_revoked = True
        await log_event("KEY_ROTATED", {"app_name": app_name, "old_key_hash_preview": key_hash[:8] + "..."})
        logger.info(f"Rotated API Key for app: {app_name}")
    except Exception as e:
        logger.error(
            f"Rotation stored a new key for app {app_name} but failed to revoke the old one ({key_hash[:8]}...): {e}"
        )
    return {
        "app_name": app_name,
        "api_key": full_key,
        "models": entry.get("models") or [],
        "default_model": entry.get("default_model"),
        "revoked_key_hash": key_hash,
        "old_key_revoked": old_key_revoked,
        "message": (
            "Store this key safely. It will not be shown again. The old key is revoked."
            if old_key_revoked
            else "Store this key safely. It will not be shown again. WARNING: revoking the old key failed — revoke it manually by hash."
        ),
    }


async def revoke_api_key_by_hash(key_hash: str) -> dict:
    deleted = await remove_api_key_by_hash(key_hash)
    if not deleted:
        raise HTTPException(status_code=404, detail="API Key not found.")
    await log_event("KEY_REVOKED", {"key_hash_preview": key_hash[:8] + "..."})
    logger.info("Revoked an API Key by hash.")
    return {"status": "success", "message": "API Key permanently revoked."}


async def get_model_registry() -> dict:
    """The configured registry, as an operator sees it.

    Unlike the app-facing ``/general-requests/models`` this is unscoped and
    includes the backend each model sits on and the pool it draws from — it is
    what the admin panel needs to render model scoping and per-pool capacity.

    ``models`` is what this process is actually serving; ``file`` is the raw
    document on disk (``null`` when there is none, meaning the registry is
    synthesized from the env vars) and ``file_error`` says why it could not be
    read when that happens. They differ exactly when the file has been edited
    and the engine has not been restarted, which ``restart_required`` reports so
    the panel can say so rather than implying a save took effect.
    """
    state = await asyncio.to_thread(registry_file_state)
    return {
        "default": default_model().id,
        "roles": {role: resolve_role(role).id for role in ROLES},
        "models": [
            {
                "id": spec.id,
                "model": spec.model,
                "api_url": spec.api_url,
                "context_window": spec.context_window,
                "pool": spec.pool,
            }
            for spec in list_models()
        ],
        "file": state["file"],
        # Distinct from a missing file: an editor must refuse to overwrite a
        # file it could not read.
        "file_error": state["error"],
        "restart_required": not state["matches_running"],
        # Whether a save can succeed at all (the file is often mounted :ro).
        "writable": state["writable"],
    }


async def save_model_registry(raw: dict | None) -> dict:
    """Validates and writes ``models.yaml``. Does NOT affect this process.

    The running registry — and the GPU pools derived from it at import — stay
    exactly as they are until every process restarts, which is the only way a
    multi-worker deployment can change registry without its workers disagreeing
    with each other mid-request.
    """
    try:
        await asyncio.to_thread(write_registry_file, raw)
    except ModelConfigError as e:
        # The document was rejected whole, so nothing was written.
        raise HTTPException(status_code=400, detail=str(e))
    except OSError as e:
        logger.error(f"Could not write the registry file: {e}")
        raise HTTPException(
            status_code=500,
            detail="Could not write models.yaml — check that the file is mounted writable.",
        )
    await log_event("MODELS_UPDATED", {"models": [m.get("id") for m in (raw or {}).get("models", [])]})
    logger.info("Model registry file updated; restart required for it to take effect.")
    state = await asyncio.to_thread(registry_file_state)
    return {
        "status": "success",
        "restart_required": not state["matches_running"],
        "detail": "Saved. Restart the engine for the new registry to take effect.",
    }


async def get_app_usage(app_name: str) -> dict:
    """Lifetime totals for one app, with the per-model split alongside them."""
    usage, by_model = await asyncio.gather(get_usage(app_name), get_usage_by_model(app_name))
    return {**usage, "by_model": by_model}


async def get_app_daily_usage(app_name: str, days: int = 7) -> dict:
    return {"app_name": app_name, "days": await get_daily_usage(app_name, days=days)}


async def get_all_usage() -> dict:
    app_names = await get_all_app_names()
    usages = await asyncio.gather(*[get_usage(name) for name in app_names])
    return {"apps": list(usages)}


async def get_gpu() -> dict:
    return await get_gpu_status()


async def reset_gpu() -> dict:
    result = await reset_gpu_counter()
    await log_event("GPU_RESET", {"reason": "manual admin reset"})
    return result


async def get_global_audit(limit: int = 100, offset: int = 0) -> dict:
    events = await get_audit_log(app_name=None, limit=limit, offset=offset)
    return {"total_returned": len(events), "events": events}


async def get_app_audit(app_name: str, limit: int = 100, offset: int = 0) -> dict:
    events = await get_audit_log(app_name=app_name, limit=limit, offset=offset)
    return {"app_name": app_name, "total_returned": len(events), "events": events}


# ── Vector DB admin ───────────────────────────────────────────────────────────

async def admin_list_all_collections() -> dict:
    collections = await get_vector_store().all_collections_with_counts()
    total_chunks = sum(c["chunk_count"] for c in collections)
    return {"total_collections": len(collections), "total_chunks": total_chunks, "collections": collections}


async def admin_list_collection_files(app_name: str, collection_name: str) -> dict:
    try:
        files = await get_vector_store().list_files(collection_name=collection_name, app_name=app_name)
        return {"app_name": app_name, "collection_name": collection_name, "files": sorted(files)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


async def admin_delete_collection(app_name: str, collection_name: str) -> dict:
    success = await get_vector_store().delete_collection(collection_name=collection_name, app_name=app_name)
    if not success:
        raise HTTPException(status_code=404, detail="Collection not found.")
    await log_event("COLLECTION_DELETED", {"collection": collection_name}, app_name=app_name)
    logger.info(f"Admin deleted collection '{collection_name}' for app '{app_name}'")
    return {"status": "success", "message": f"Collection '{collection_name}' deleted."}


async def admin_vector_search(app_name: str, collection_name: str, query: str, n_results: int = 5) -> dict:
    store = get_vector_store()
    try:
        results = await store.search(
            collection_name=collection_name, app_name=app_name, query=query, n_results=n_results
        )
        # The score scale differs by backend: hybrid (pgvector) returns small RRF
        # scores, non-hybrid (chroma) returns a 0–1 similarity. Tell the UI which
        # so it can colour the score badge against the right thresholds.
        return {
            "query": query,
            "app_name": app_name,
            "collection_name": collection_name,
            "results": results,
            "score_type": "rrf" if store.supports_hybrid else "similarity",
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


async def admin_delete_file(app_name: str, collection_name: str, filename: str) -> dict:
    try:
        await get_vector_store().delete_file(collection_name=collection_name, filename=filename, app_name=app_name)
        await log_event("FILE_DELETED", {"filename": filename, "collection": collection_name}, app_name=app_name)
        logger.info(f"Admin deleted file '{filename}' from '{collection_name}' for app '{app_name}'")
        return {"status": "success", "message": f"File '{filename}' deleted."}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
