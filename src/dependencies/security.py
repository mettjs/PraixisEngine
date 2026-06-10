import asyncio
import secrets

from fastapi import Depends, Security, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.security.api_key import APIKeyHeader
from src.config import ADMIN_USERNAME, ADMIN_PASSWORD
from src.utils.store.api_keys import lookup_api_key
from src.utils.store.audit import log_event
from src.utils.system.logger import logger

_API_KEY_NAME = "X-API-Key"
_api_key_header = APIKeyHeader(name=_API_KEY_NAME, auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)) -> str:
    """Checks Redis for the API Key (by SHA-256 hash) and returns the associated App Name."""
    if not api_key:
        logger.warning("Request done without an API Key.")
        raise HTTPException(status_code=403, detail="API Key header missing.")

    app_name = await lookup_api_key(api_key)

    if not app_name:
        logger.warning("Invalid API Key attempted.")
        await log_event("AUTH_FAIL", {"key_preview": api_key[:14] + "..." if len(api_key) > 14 else "***"})
        # Auth failures never reach the per-key rate limiter (it wraps the
        # endpoint, which runs after this dependency), so throttle key-guessing
        # and the Redis lookups + audit writes it generates here.
        await asyncio.sleep(0.5)
        raise HTTPException(status_code=403, detail="Invalid or revoked API Key.")

    logger.info(f"API Key authenticated for app: {app_name}")
    return app_name


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
