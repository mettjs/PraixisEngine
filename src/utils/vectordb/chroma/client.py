"""Chroma client lifecycle, tenant scoping, and ownership checks.

Collections are scoped per app by name (``{app}.{collection}``) and stamped
with ``metadata["app"]`` for ownership verification — names alone are not
trusted. The ``.`` separator cannot appear in app or collection names
(both are restricted to ``[a-zA-Z0-9_-]``), so the scoped name is
unambiguous; the older ``_`` separator was not and is migrated at startup.
The hypothetical-question index lives in a parallel collection
(``{app}.{collection}__hyq``) flagged with ``metadata["hyq"]`` so listings can
exclude it by metadata instead of name parsing.

Everything here is synchronous (Chroma's client is); callers wrap whole
operations in ``asyncio.to_thread``.
"""
import chromadb
from chromadb.config import Settings

from src.config import CHROMA_PATH as _CHROMA_PATH
from src.utils.system.logger import logger

_client: chromadb.ClientAPI | None = None

_QUESTIONS_SUFFIX = "__hyq"

# Chroma rejects collection names longer than 63 characters.
_MAX_NAME_LEN = 63


def init_client() -> None:
    global _client
    _client = chromadb.PersistentClient(
        path=_CHROMA_PATH,
        # A self-hosted backend should not phone home (and the posthog client
        # logs errors on every event when the host has no outbound network).
        settings=Settings(anonymized_telemetry=False),
    )
    _migrate_legacy_names(_client)


def _migrate_legacy_names(client: chromadb.ClientAPI) -> None:
    """One-time rename of legacy ``{app}_{collection}`` names to the
    ``{app}.{collection}`` scheme.

    The old ``_`` separator was ambiguous — app and collection names may
    themselves contain underscores, so two different (app, collection) pairs
    could collide on one scoped name. ``metadata["app"]`` has always been
    stamped at creation, so it recovers the split unambiguously. New-scheme
    names never start with ``{app}_`` (they start with ``{app}.``), making
    this a no-op once everything is migrated.
    """
    for col in client.list_collections():
        app = (col.metadata or {}).get("app")
        if not isinstance(app, str) or not col.name.startswith(f"{app}_"):
            continue
        new_name = f"{app}.{col.name[len(app) + 1:]}"
        try:
            col.modify(name=new_name)
            logger.info(f"Migrated Chroma collection '{col.name}' -> '{new_name}'.")
        except Exception as e:
            logger.error(f"Failed to migrate Chroma collection '{col.name}': {e}")


def close_client() -> None:
    # PersistentClient flushes on every write; there is no close handle.
    global _client
    _client = None


def get_client() -> chromadb.ClientAPI:
    if _client is None:
        raise RuntimeError("Chroma client not initialized.")
    return _client


def scoped_name(app_name: str, collection_name: str) -> str:
    return f"{app_name}.{collection_name}"


def questions_name(app_name: str, collection_name: str) -> str:
    return scoped_name(app_name, collection_name) + _QUESTIONS_SUFFIX


def ensure_name_fits(app_name: str, collection_name: str) -> None:
    """Raises ValueError when the scoped name would exceed Chroma's limit.

    Guards against the *questions* name (scoped name + suffix) so a collection
    that fits today can't fail later when its question index is created.
    """
    if len(questions_name(app_name, collection_name)) > _MAX_NAME_LEN:
        raise ValueError(
            f"Combined app and collection name is too long for the Chroma backend "
            f"(max {_MAX_NAME_LEN - len(_QUESTIONS_SUFFIX) - 1} characters together). "
            f"Use a shorter collection name."
        )


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


def drop_collection(collection_name: str, app_name: str) -> None:
    """Removes an app's collection along with the question index that shares its
    lifecycle.

    Callers must have verified ownership first — this does not re-check. The
    question index is skipped when absent (or not owned, which
    ``get_questions_collection`` reports as absent).
    """
    get_client().delete_collection(name=scoped_name(app_name, collection_name))
    if get_questions_collection(collection_name, app_name) is not None:
        get_client().delete_collection(name=questions_name(app_name, collection_name))


def get_questions_collection(collection_name: str, app_name: str) -> chromadb.Collection | None:
    """The parallel question index, or None when it was never created."""
    try:
        collection = get_client().get_collection(name=questions_name(app_name, collection_name))
    except Exception:
        return None
    return collection if is_owned(collection, app_name) else None
