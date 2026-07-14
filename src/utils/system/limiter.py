"""Async fixed-window rate limiting backed by Redis.

This replaces slowapi: its storage layer is synchronous, so every rate-limit
check was a blocking Redis round-trip on the event loop. Counters live in
Redis so limits are global across workers and replicas and survive restarts —
an in-memory counter would multiply every limit by the number of uvicorn
workers.
"""
import time

from fastapi import HTTPException, Request

from src.utils.store.api_keys import KEY_PREFIX, hash_api_key
from src.utils.store.client import redis_client

_PERIODS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _bucket_key(request: Request) -> str:
    """Buckets by API key for authenticated traffic, by client IP otherwise.

    Only keys carrying the issued prefix get their own bucket — keying on the
    raw header would hand a caller a fresh bucket per fabricated value,
    sidestepping the IP limit entirely. The bucket name carries the key's
    digest, never the raw secret: it is the same ``key_hash`` the admin
    endpoints show, so buckets stay attributable without exposing the key
    to anything that can read or log Redis key names.
    """
    api_key = request.headers.get("X-API-Key")
    if api_key and api_key.startswith(KEY_PREFIX):
        return hash_api_key(api_key)
    return client_ip(request)


def rate_limit(limit: str):
    """Dependency factory: ``Depends(rate_limit("10/minute"))``.

    Fixed-window counting, one INCR+EXPIRE pipeline per request, keyed by
    (route template, caller bucket, window index). Raises 429 once the
    window's count exceeds the limit.
    """
    count_str, _, period = limit.partition("/")
    max_requests = int(count_str)
    window_seconds = _PERIODS[period]

    async def _check(request: Request) -> None:
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        window = int(time.time() // window_seconds)
        key = f"ratelimit:{path}:{_bucket_key(request)}:{window}"
        pipe = redis_client.pipeline()
        pipe.incr(key)
        pipe.expire(key, window_seconds)
        current, _ = await pipe.execute()
        if int(current) > max_requests:
            raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {limit}.")

    return _check
