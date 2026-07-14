"""Single source of truth for configuration.

Every environment variable is read and parsed here exactly once. Import the
parsed values from this module instead of calling ``os.getenv`` elsewhere.
Importing this module also loads the ``.env`` file, so it is self-sufficient
regardless of import order.
"""
import os

from dotenv import find_dotenv, load_dotenv


def _load_dotenv() -> None:
    """Loads the .env file if present.

    A missing .env is fine — vars may be injected directly (Docker, k8s, CI).
    A present-but-unloadable .env is a hard error.
    """
    env_file = find_dotenv()
    if not env_file:
        return
    if not load_dotenv(env_file):
        raise RuntimeError(f"Found .env at '{env_file}' but failed to load it.")


_load_dotenv()

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --- Upload limits (constant, not env-derived) ---
# One cap for every ingestion path: multipart file uploads and raw-text bodies.
# Lives here so schema validation and the file parser share a single value
# without the schemas module having to import the parser's heavy dependencies.
MAX_FILE_SIZE: int = 20 * 1024 * 1024  # bytes

# --- LLM backend ---
AI_API_URL: str = os.getenv("AI_API_URL", "http://localhost:8081")
AI_API_KEY: str = os.getenv("AI_API_KEY", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gemma-api-test")

# --- Redis & sessions ---
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL: int = int(os.getenv("SESSION_TTL", "86400"))          # seconds

# Token budget for a session's conversation history (history size is estimated
# at ~4 chars/token). When a session approaches this budget, older exchanges are
# automatically compacted into an LLM-written summary instead of being dropped.
# Size it to your model's context window, leaving room for the completion and,
# on RAG requests, the retrieved context.
CONTEXT_WINDOW: int = int(os.getenv("CONTEXT_WINDOW", "8192"))

# --- GPU concurrency ---
GPU_CONCURRENCY: int = int(os.getenv("GPU_CONCURRENCY", "2"))
GPU_WAIT_TIMEOUT: float = float(os.getenv("GPU_WAIT_TIMEOUT", "30"))
CHUNK_CONCURRENCY: int = int(os.getenv("CHUNK_CONCURRENCY", "4"))

# Slots reserved exclusively for background hypothetical-question generation.
# This is a SEPARATE pool from GPU_CONCURRENCY, so question generation can never
# starve interactive chat/RAG and is never starved by it; total concurrent LLM
# calls a deployment can issue is GPU_CONCURRENCY + HQ_GPU_CONCURRENCY. Set to 0
# to disable the dedicated pool (generation then falls back to the shared pool).
# Because generation is background and best-effort, it waits much longer for a
# slot than interactive requests before giving up on a chunk.
HQ_GPU_CONCURRENCY: int = int(os.getenv("HQ_GPU_CONCURRENCY", "1"))
HQ_GPU_WAIT_TIMEOUT: float = float(os.getenv("HQ_GPU_WAIT_TIMEOUT", "300"))

# --- Vector store ---
# 'pgvector' (Postgres: hybrid dense+FTS retrieval, FK-backed question index)
# or 'chroma' (embedded ChromaDB: no extra database to run, pure vector search).
VECTOR_BACKEND: str = os.getenv("VECTOR_BACKEND", "pgvector").strip().lower()
POSTGRES_URL: str = os.getenv("POSTGRES_URL", "postgresql://praixis:praixis@localhost:5432/praixis")
CHROMA_PATH: str = os.getenv("CHROMA_PATH", os.path.join(_ROOT, "chroma_data"))
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_DIMS: int = int(os.getenv("EMBEDDING_DIMS", "384"))

# --- Hypothetical-question indexing ---
# After a file's chunks are stored, an LLM generates civilian-register questions
# each chunk answers; those questions are embedded and indexed for question-to-
# question retrieval. Runs as a deferred background pass, so it never blocks the
# upload response. Set HQ_ENABLED=false to skip generation entirely.
HQ_ENABLED: bool = os.getenv("HQ_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
HQ_PER_CHUNK: int = int(os.getenv("HQ_PER_CHUNK", "5"))

# --- Admin auth (no defaults — must be set in the environment) ---
ADMIN_USERNAME: str | None = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD: str | None = os.getenv("ADMIN_PASSWORD")
