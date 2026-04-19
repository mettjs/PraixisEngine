from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

def _get_rate_limit_key(request: Request) -> str:
    """Rate-limits by API key when present, falling back to IP for unauthenticated routes."""
    api_key = request.headers.get("X-API-Key")
    return api_key if api_key else get_remote_address(request)

limiter = Limiter(key_func=_get_rate_limit_key)