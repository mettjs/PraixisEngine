import asyncio
import secrets

from fastapi import Depends, Request, Security, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.security.api_key import APIKeyHeader
from src.config import ADMIN_USERNAME, ADMIN_PASSWORD
from src.utils.store.api_keys import lookup_api_key_entry
from src.utils.store.audit import log_event
from src.utils.store.client import redis_client
from src.utils.system.limiter import client_ip
from src.utils.system.logger import logger

_API_KEY_NAME = "X-API-Key"
_api_key_header = APIKeyHeader(name=_API_KEY_NAME, auto_error=False)

# Invalid-key attempts allowed per client IP per window. The 0.5s sleep below
# only bounds per-request latency, not concurrency — a parallel brute force
# could still probe keys at scale without this hard cap.
_AUTH_FAIL_LIMIT = 30
_AUTH_FAIL_WINDOW = 60  # seconds


async def verify_api_key(request: Request, api_key: str = Security(_api_key_header)) -> str:
    """Checks Redis for the API Key (by SHA-256 hash) and returns the associated App Name.

    The whole entry — including any model scoping — is stashed on
    ``request.state.caller`` for the few endpoints that need it (see
    :func:`caller`). Returning just the app name keeps every existing
    ``Depends(verify_api_key)`` signature untouched, and the entry is already
    in hand, so nothing costs an extra Redis round-trip.
    """
    if not api_key:
        logger.warning("Request done without an API Key.")
        raise HTTPException(status_code=403, detail="API Key header missing.")

    entry = await lookup_api_key_entry(api_key)
    app_name = entry.get("app_name") if entry else None

    if not app_name:
        fail_key = f"authfail:{client_ip(request)}"
        pipe = redis_client.pipeline()
        pipe.incr(fail_key)
        pipe.expire(fail_key, _AUTH_FAIL_WINDOW)
        fails, _ = await pipe.execute()
        if int(fails) > _AUTH_FAIL_LIMIT:
            raise HTTPException(status_code=429, detail="Too many failed authentication attempts. Try again later.")
        logger.warning("Invalid API Key attempted.")
        await log_event("AUTH_FAIL", {"key_preview": api_key[:14] + "..." if len(api_key) > 14 else "***"})
        # Auth failures never reach the per-key rate limiter (it only counts
        # authenticated requests), so also slow each attempt and the Redis
        # lookups + audit writes it generates.
        await asyncio.sleep(0.5)
        raise HTTPException(status_code=403, detail="Invalid or revoked API Key.")

    request.state.caller = entry
    logger.info(f"API Key authenticated for app: {app_name}")
    return app_name


def caller(request: Request) -> dict:
    """The authenticated key's stored entry, for endpoints that need more than
    the app name (its ``models`` allowlist and ``default_model``).

    Depends on nothing: ``verify_api_key`` already put it on the request, and
    every route reaching here is behind that dependency.
    """
    return getattr(request.state, "caller", None) or {}


_security_basic = HTTPBasic()


async def verify_admin_credentials(credentials: HTTPBasicCredentials = Depends(_security_basic)):
    """Validates the master username and password from the .env file."""
    correct_username = ADMIN_USERNAME
    correct_password = ADMIN_PASSWORD

    if not (correct_username and correct_password):
        logger.warning("Master Username and Password not found.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Master Username and Password not found.",
            headers={"WWW-Authenticate": "Basic"},
        )

    is_correct_username = secrets.compare_digest(
        credentials.username.encode("utf8"), correct_username.encode("utf8")
    )
    is_correct_password = secrets.compare_digest(
        credentials.password.encode("utf8"), correct_password.encode("utf8")
    )

    if not (is_correct_username and is_correct_password):
        logger.warning("Incorrect Master Username or Password")
        # Basic auth has no lockout, so slow brute-force attempts down. The
        # sleep is async — it ties up this request, not a worker.
        await asyncio.sleep(1.0)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect Master Username or Password",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username
