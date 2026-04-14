from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from src.utils.limiter import limiter
from slowapi.extension import _rate_limit_exceeded_handler
from src.routes.main_router import api_router
from src.utils.load_env import load_env

try: 
    load_env()
except Exception as e:
    raise RuntimeError(f"ERROR: Could not load .env file: {e}")

app = FastAPI(
    title="Praixis - Business logic based API",
    description="Custom decoupled business logic API powered by Gemma 4",
    version="1.0.0",
    redoc_url="/docs",
)

app.state.limiter = limiter

# Tell FastAPI to return a clean 429 error when someone is blocked
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler) # type: ignore[arg-type]

app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)