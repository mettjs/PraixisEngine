import uuid
import os
import json
import re
import redis
from typing import List, Dict, Tuple

_REDIS = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(_REDIS, decode_responses=True)

_SESSION_TTL: int = int(os.getenv("SESSION_TTL", 86400)) # 24 hours is the default expiration time for sessions if not set in .env

def _get_redis_key(app_name: str, session_id: str) -> str:
    """Helper to ensure perfectly consistent key formatting."""
    return f"chat:{app_name}:{session_id}"

def get_or_create_session(
    app_name: str,
    session_id: str | None = None, 
    system_prompt: str | None = None
) -> Tuple[str, List[Dict[str, str]]]:
    
    if not session_id or not re.fullmatch(r"[0-9a-f]{32}", session_id):
        session_id = None
        
    final_prompt = system_prompt or "You are a helpful institutional assistant."
    
    if session_id:
        redis_key = _get_redis_key(app_name, session_id)
        stored_data = redis_client.get(redis_key)
        
        if isinstance(stored_data, str):
            history = json.loads(stored_data)
            
            if system_prompt and len(history) > 0 and history[0].get("role") == "system":
                history[0]["content"] = system_prompt
                redis_client.setex(redis_key, _SESSION_TTL, json.dumps(history))
            else:
                redis_client.expire(redis_key, _SESSION_TTL)
                
            return session_id, history
            
    new_session_id = uuid.uuid4().hex
    new_redis_key = _get_redis_key(app_name, new_session_id)
    initial_history = [{"role": "system", "content": final_prompt}]
    
    redis_client.setex(new_redis_key, _SESSION_TTL, json.dumps(initial_history))
    
    return new_session_id, initial_history

def add_message(app_name: str, session_id: str, role: str, content: str):
    redis_key = _get_redis_key(app_name, session_id)
    stored_data = redis_client.get(redis_key)
    
    if isinstance(stored_data, str):
        history = json.loads(stored_data)
        history.append({"role": role, "content": content})
        redis_client.setex(redis_key, _SESSION_TTL, json.dumps(history))

def get_session_history(app_name: str, session_id: str) -> list:
    redis_key = _get_redis_key(app_name, session_id)
    data = redis_client.get(redis_key)
    if isinstance(data, str):
        return json.loads(data)
    return []

def delete_session(app_name: str, session_id: str) -> bool:
    redis_key = _get_redis_key(app_name, session_id)
    return redis_client.delete(redis_key) > 0  # type: ignore[operator]

def get_all_active_sessions(app_name: str) -> list:
    """Returns only the session IDs for THIS specific app."""
    pattern = f"chat:{app_name}:*"
    raw_keys = list(redis_client.scan_iter(pattern))  # type: ignore[arg-type, return-value]
    
    prefix_length = len(f"chat:{app_name}:")
    clean_uuids = [key[prefix_length:] for key in raw_keys]
    
    return clean_uuids