try:
    from src import config  # noqa: F401 — loads .env and parses settings (single source of truth)
except Exception as e:
    raise RuntimeError(f"ERROR: Could not load configuration: {e}")

from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.routes.main_router import api_router
from src.routes.ui_router import STATIC_DIR


def _engine_version() -> str:
    """pyproject.toml is the single source of truth for the version: installed
    deployments read it from package metadata, source checkouts fall back to
    parsing the file directly."""
    try:
        return _pkg_version("praixisengine")
    except PackageNotFoundError:
        import tomllib
        from pathlib import Path
        with (Path(__file__).parent / "pyproject.toml").open("rb") as f:
            return tomllib.load(f)["project"]["version"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.utils.concurrency import init_gpu
    from src.utils.vectordb import get_vector_store
    store = get_vector_store()  # resolves VECTOR_BACKEND (pgvector | chroma)
    await init_gpu()
    await store.init()
    yield
    await store.close()


app = FastAPI(
    title="Praixis - Business logic based API",
    description="Custom decoupled business logic API powered by a local OpenAI-compatible LLM.",
    version=_engine_version(),
    docs_url="/swagger/docs",
    redoc_url="/docs",
    lifespan=lifespan,
)

app.include_router(api_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
