from fastapi import APIRouter, Depends
from src.dependencies.security import verify_admin_credentials
from src.controllers.admin_controller import (
    generate_api_key,
    get_all_usage,
    get_app_usage,
    get_health_status,
    get_system_stats,
    list_api_keys,
    revoke_api_key,
    delete_app_sessions,
)

router = APIRouter(
    prefix="/api/system",
    tags=["System Admin"],
    dependencies=[Depends(verify_admin_credentials)]
)


@router.get("/health")
async def system_health_check():
    return await get_health_status()


@router.get("/stats")
async def system_statistics():
    return await get_system_stats()


@router.get("/keys")
async def list_keys():
    return await list_api_keys()


@router.post("/keys/generate")
async def create_app_key(app_name: str):
    return await generate_api_key(app_name)


@router.delete("/keys/revoke")
async def delete_app_key(api_key: str):
    return await revoke_api_key(api_key)


@router.delete("/sessions/{app_name}")
async def wipe_sessions(app_name: str):
    return await delete_app_sessions(app_name)


@router.get("/usage")
async def all_usage():
    return await get_all_usage()


@router.get("/usage/{app_name}")
async def app_usage(app_name: str):
    return await get_app_usage(app_name)
