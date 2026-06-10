from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.config import REDIS_URL

def _get_rate_limit_key(request: Request) -> str:
    """Rate-limits by API key when present, falling back to IP for unauthenticated routes."""
    api_key = request.headers.get("X-API-Key")
    return api_key if api_key else get_remote_address(request)

# Counters live in Redis so limits are global across workers and replicas and
# survive restarts — the in-memory default would multiply every limit by the
# number of uvicorn workers.
limiter = Limiter(key_func=_get_rate_limit_key, storage_uri=REDIS_URL)