from fastapi import APIRouter, Depends
from src.dependencies.security import verify_admin_credentials
from src.controllers.admin_controller import generate_api_key, get_health_status, get_system_stats, revoke_api_key

router = APIRouter(
    prefix="/api/system", 
    tags=["System Admin"],
    dependencies=[Depends(verify_admin_credentials)]
)

@router.get("/health")
def system_health_check():
    return get_health_status()

@router.get("/stats")
def system_statistics():
    return get_system_stats()

@router.post("/keys/generate")
def create_app_key(app_name: str):
    return generate_api_key(app_name)

@router.delete("/keys/revoke")
def delete_app_key(api_key: str):
    return revoke_api_key(api_key)