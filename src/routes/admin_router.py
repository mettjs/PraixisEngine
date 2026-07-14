from fastapi import APIRouter, Depends, Query
from src.dependencies.security import verify_admin_credentials
from src.controllers.admin_controller import (
    generate_api_key,
    get_all_usage,
    get_app_usage,
    get_app_daily_usage,
    rotate_api_key,
    get_health_status,
    get_redis_health,
    get_vectordb_health,
    get_llm_health,
    get_system_stats,
    list_api_keys,
    revoke_api_key_by_hash,
    delete_app_sessions,
    get_gpu,
    reset_gpu,
    get_global_audit,
    get_app_audit,
    admin_list_all_collections,
    admin_list_collection_files,
    admin_delete_collection,
    admin_delete_file,
    admin_vector_search,
)

router = APIRouter(
    prefix="/api/system",
    tags=["System Admin"],
    dependencies=[Depends(verify_admin_credentials)]
)

# Public router — no auth. Holds the liveness check used by container healthchecks.
public_router = APIRouter(prefix="/api/system", tags=["System Admin"])


@public_router.get("/ping")
async def admin_ping():
    return {"ok": True}


@router.get("/auth/verify")
async def verify_admin_auth():
    """Authenticated no-op the admin panel calls to validate credentials."""
    return {"ok": True}


@router.get("/health")
async def system_health_check():
    return await get_health_status()


@router.get("/health/redis")
async def redis_health():
    return await get_redis_health()


@router.get("/health/vectordb")
async def vectordb_health():
    return await get_vectordb_health()


@router.get("/health/llm")
async def llm_health():
    return await get_llm_health()


@router.get("/stats")
async def system_statistics():
    return await get_system_stats()


@router.get("/keys")
async def list_keys():
    return await list_api_keys()


@router.post("/keys/generate")
async def create_app_key(app_name: str = Query(..., pattern=r"^[a-zA-Z0-9_-]{3,63}$")):
    return await generate_api_key(app_name)


@router.post("/keys/rotate")
async def rotate_app_key(key_hash: str = Query(..., description="Hash of the key to replace (see GET /keys).")):
    """Issues a new key for the app owning ``key_hash``, then revokes the old
    one — the app never has a window with zero valid keys."""
    return await rotate_api_key(key_hash)


@router.delete("/keys/revoke-by-hash")
async def delete_app_key_by_hash(key_hash: str):
    return await revoke_api_key_by_hash(key_hash)


@router.delete("/sessions/{app_name}")
async def wipe_sessions(app_name: str):
    return await delete_app_sessions(app_name)


@router.get("/usage")
async def all_usage():
    return await get_all_usage()


@router.get("/usage/{app_name}")
async def app_usage(app_name: str):
    return await get_app_usage(app_name)


@router.get("/usage/{app_name}/daily")
async def app_usage_daily(
    app_name: str,
    days: int = Query(default=7, ge=1, le=90, description="How many UTC days back to report, most recent first."),
):
    """Per-day token/request counts (kept 90 days), alongside the lifetime totals."""
    return await get_app_daily_usage(app_name, days=days)


@router.get("/gpu")
async def gpu_status():
    return await get_gpu()


@router.post("/gpu/reset")
async def reset_gpu_slots():
    return await reset_gpu()


@router.get("/audit")
async def global_audit(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    return await get_global_audit(limit=limit, offset=offset)


@router.get("/audit/{app_name}")
async def app_audit(
    app_name: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    return await get_app_audit(app_name=app_name, limit=limit, offset=offset)


@router.get("/vector/search")
async def vector_search(
    app_name: str,
    collection_name: str,
    query: str,
    n_results: int = Query(default=5, ge=1, le=20),
):
    return await admin_vector_search(app_name=app_name, collection_name=collection_name, query=query, n_results=n_results)


@router.get("/vector/collections")
async def vector_collections():
    return await admin_list_all_collections()


@router.get("/vector/collections/{app_name}/{collection_name}/files")
async def vector_collection_files(app_name: str, collection_name: str):
    return await admin_list_collection_files(app_name=app_name, collection_name=collection_name)


@router.delete("/vector/collections/{app_name}/{collection_name}")
async def vector_delete_collection(app_name: str, collection_name: str):
    return await admin_delete_collection(app_name=app_name, collection_name=collection_name)


@router.delete("/vector/collections/{app_name}/{collection_name}/files")
async def vector_delete_file(app_name: str, collection_name: str, filename: str = Query(...)):
    return await admin_delete_file(app_name=app_name, collection_name=collection_name, filename=filename)
