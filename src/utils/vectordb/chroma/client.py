"""Chroma client lifecycle, tenant scoping, and ownership checks.

Collections are scoped per app by name (``{app}_{collection}``) and stamped
with ``metadata["app"]`` for ownership verification — names alone are not
trusted. The hypothetical-question index lives in a parallel collection
(``{app}_{collection}__hyq``) flagged with ``metadata["hyq"]`` so listings can
exclude it by metadata instead of name parsing.

Everything here is synchronous (Chroma's client is); callers wrap whole
operations in ``asyncio.to_thread``.
"""
import chromadb
from chromadb.config import Settings

from src.config import CHROMA_PATH as _CHROMA_PATH

_client: chromadb.ClientAPI | None = None

_QUESTIONS_SUFFIX = "__hyq"


def init_client() -> None:
    global _client
    _client = chromadb.PersistentClient(
        path=_CHROMA_PATH,
        # A self-hosted backend should not phone home (and the posthog client
        # logs errors on every event when the host has no outbound network).
        settings=Settings(anonymized_telemetry=False),
    )


def close_client() -> None:
    # PersistentClient flushes on every write; there is no close handle.
    global _client
    _client = None


def get_client() -> chromadb.ClientAPI:
    if _client is None:
        raise RuntimeError("Chroma client not initialized.")
    return _client


def scoped_name(app_name: str, collection_name: str) -> str:
    return f"{app_name}_{collection_name}"


def questions_name(app_name: str, collection_name: str) -> str:
    return scoped_name(app_name, collection_name) + _QUESTIONS_SUFFIX


def is_owned(collection: chromadb.Collection, app_name: str) -> bool:
    return bool(collection.metadata) and collection.metadata.get("app") == app_name


def get_owned_collection(collection_name: str, app_name: str) -> chromadb.Collection:
    """The app's chunks collection. Raises ValueError when missing or not owned."""
    try:
        collection = get_client().get_collection(name=scoped_name(app_name, collection_name))
    except Exception:
        raise ValueError(f"The collection '{collection_name}' does not exist.")
    if not is_owned(collection, app_name):
        raise ValueError("Access denied: You do not own this collection.")
    return collection


def get_questions_collection(collection_name: str, app_name: str) -> chromadb.Collection | None:
    """The parallel question index, or None when it was never created."""
    try:
        collection = get_client().get_collection(name=questions_name(app_name, collection_name))
    except Exception:
        return None
    return collection if is_owned(collection, app_name) else None
