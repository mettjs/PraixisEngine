import asyncio
import secrets
from fastapi import HTTPException
from src.utils.memory import redis_client, delete_all_app_sessions, get_usage, get_all_app_names
from src.utils.vector_db import chroma_client
from src.utils.ai_client import get_ai_client
from src.utils.logger import logger

# Sync client used only for the health-check ping (no LLM calls, no token tracking)
_llm_sync_client = get_ai_client()


async def get_health_status() -> dict:
    health_status = {"api": "online", "redis": "offline", "chromadb": "offline", "llm": "offline"}

    try:
        await redis_client.ping()  # type: ignore[misc]
        health_status["redis"] = "online"
    except Exception:
        logger.error("Redis health check failed.")

    try:
        await asyncio.to_thread(chroma_client.list_collections)
        health_status["chromadb"] = "online"
    except Exception:
        logger.error("ChromaDB health check failed.")

    try:
        await asyncio.to_thread(lambda: _llm_sync_client.with_options(timeout=5.0).models.list())
        health_status["llm"] = "online"
    except Exception:
        logger.error("LLM backend health check failed.")

    return health_status


async def get_system_stats() -> dict:
    active_sessions = 0
    async for _ in redis_client.scan_iter("chat:*"):
        active_sessions += 1

    collections = await asyncio.to_thread(chroma_client.list_collections)
    total_vectors = sum([col.count() for col in collections])

    return {
        "active_chat_sessions": active_sessions,
        "total_vector_collections": len(collections),
        "total_vector_chunks": total_vectors,
    }


async def generate_api_key(app_name: str) -> dict:
    raw_key = secrets.token_urlsafe(32)
    full_key = f"praxis_{raw_key}"
    await redis_client.set(f"apikey:{full_key}", app_name)
    logger.info(f"Generated new API Key for app: {app_name}")
    return {"app_name": app_name, "api_key": full_key, "message": "Store this key safely. It will not be shown again."}


async def revoke_api_key(api_key: str) -> dict:
    app_name = await redis_client.get(f"apikey:{api_key}")
    deleted = await redis_client.delete(f"apikey:{api_key}")

    if not deleted:
        raise HTTPException(status_code=404, detail="API Key not found.")

    logger.info(f"Revoked API Key for app: {app_name}")
    return {"status": "success", "message": "API Key permanently revoked."}


async def list_api_keys() -> dict:
    """Lists all provisioned API keys with masked values."""
    entries = []
    async for redis_key in redis_client.scan_iter("apikey:*"):
        app_name = await redis_client.get(redis_key)
        full_key = str(redis_key).removeprefix("apikey:")
        masked = full_key[:14] + "..." if len(full_key) > 14 else full_key
        entries.append({"app_name": str(app_name), "api_key_preview": masked})
    return {"total_keys": len(entries), "keys": entries}


async def delete_app_sessions(app_name: str) -> dict:
    """Force-expires all Redis sessions belonging to a specific app."""
    count = await delete_all_app_sessions(app_name)
    logger.info(f"Wiped {count} session(s) for app: {app_name}")
    return {"status": "success", "sessions_deleted": count, "app_name": app_name}


async def get_app_usage(app_name: str) -> dict:
    return await get_usage(app_name)


async def get_all_usage() -> dict:
    app_names = await get_all_app_names()
    return {"apps": [await get_usage(name) for name in app_names]}
