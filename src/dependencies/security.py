import os
import secrets

from fastapi import Depends, Security, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.security.api_key import APIKeyHeader
from src.utils.memory import redis_client
from src.utils.logger import logger

# The header name the client apps must use
_API_KEY_NAME = "X-API-Key"
_api_key_header = APIKeyHeader(name=_API_KEY_NAME, auto_error=False)

def verify_api_key(api_key: str = Security(_api_key_header)) -> str:
    """Checks Redis for the API Key and returns the associated App Name."""
    if not api_key:
        logger.warning("Request done without an API Key.")
        raise HTTPException(status_code=403, detail="API Key header missing.")
        
    # We prefix keys with 'apikey:' in Redis to keep them organized
    app_name = redis_client.get(f"apikey:{api_key}")
    
    if not app_name:
        logger.warning(f"Invalid API Key attempted: {api_key}")
        raise HTTPException(status_code=403, detail="Invalid or revoked API Key.")
    
    logger.info(f"API Key authenticated for app: {app_name}")
        
    return str(app_name)

_security_basic = HTTPBasic()

def verify_admin_credentials(credentials: HTTPBasicCredentials = Depends(_security_basic)):
    """Validates the master username and password from the .env file."""
    
    correct_username = os.getenv("ADMIN_USERNAME")
    correct_password = os.getenv("ADMIN_PASSWORD")
    
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
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect Master Username or Password",
            # This specific header triggers the browser's native login popup!
            headers={"WWW-Authenticate": "Basic"},
        )
        
    return credentials.username