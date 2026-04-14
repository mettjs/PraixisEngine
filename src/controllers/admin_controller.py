import secrets
from fastapi import HTTPException
from src.utils.memory import redis_client
from src.utils.vector_db import chroma_client
from src.utils.logger import logger

def get_health_status() -> dict:
    health_status = {"api": "online", "redis": "offline", "chromadb": "offline"}
    
    try:
        if redis_client.ping():
            health_status["redis"] = "online"
    except Exception:
        logger.error("Redis health check failed.")
        pass
        
    try:
        chroma_client.list_collections()
        health_status["chromadb"] = "online"
    except Exception:
        logger.error("ChromaDB health check failed.")
        pass
        
    return health_status

def get_system_stats() -> dict:
    active_sessions = sum(1 for _ in redis_client.scan_iter("chat:*")) # type: ignore[arg-type]
    collections = chroma_client.list_collections()
    total_vectors = sum([col.count() for col in collections])
    
    return {
        "active_chat_sessions": active_sessions,
        "total_vector_collections": len(collections),
        "total_vector_chunks": total_vectors
    }
    
def generate_api_key(app_name: str) -> dict:
    raw_key = secrets.token_urlsafe(32)
    full_key = f"praxis_{raw_key}"
    
    redis_client.set(f"apikey:{full_key}", app_name)
    
    logger.info(f"Generated new API Key for app: {app_name}")
    
    return {
        "app_name": app_name, 
        "api_key": full_key, 
        "message": "Store this key safely. It will not be shown again."
    }

def revoke_api_key(api_key: str) -> dict:
    app_name = redis_client.get(f"apikey:{api_key}")
    deleted = redis_client.delete(f"apikey:{api_key}") # type: ignore[operator]
    
    if not deleted:
        raise HTTPException(status_code=404, detail="API Key not found.")
        
    logger.info(f"Revoked API Key for app: {app_name}")
    
    return {"status": "success", "message": "API Key permanently revoked."}