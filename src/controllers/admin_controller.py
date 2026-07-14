import asyncio
from fastapi import HTTPException
from src.utils.store.client import redis_client
from src.utils.store.sessions import delete_all_app_sessions
from src.utils.store.usage import get_usage, get_all_app_names, get_daily_usage
from src.utils.store.api_keys import (
    store_api_key,
    remove_api_key_by_hash,
    list_all_api_keys,
    get_api_key_entry,
    new_api_key,
)
from src.utils.vectordb import get_vector_store
from src.utils.ai_client import get_async_ai_client
from src.utils.concurrency import get_gpu_status, reset_gpu_counter
from src.utils.store.audit import log_event, get_audit_log
from src.utils.system.logger import logger

# Used only for the health-check ping (no LLM calls, no token tracking)
_llm_client = get_async_ai_client()


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
    try:
        await _llm_client.with_options(timeout=5.0).models.list()
        return {"status": "online"}
    except Exception:
        logger.error("LLM backend health check failed.")
        return {"status": "offline"}


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


async def generate_api_key(app_name: str) -> dict:
    full_key = new_api_key()
    await store_api_key(full_key, app_name)
    await log_event("KEY_GENERATED", {"app_name": app_name})
    logger.info(f"Generated new API Key for app: {app_name}")
    return {"app_name": app_name, "api_key": full_key, "message": "Store this key safely. It will not be shown again."}


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
    await store_api_key(full_key, app_name)
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


async def get_app_usage(app_name: str) -> dict:
    return await get_usage(app_name)


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
