from typing import Any

from src.utils.vectordb.pool import get_pool
from src.utils.vectordb.constants import (
    ALL_COLLECTIONS_ADMIN,
    VECTOR_STATS,
    LIST_COLLECTIONS,
    LIST_FILES,
    COLLECTION_EXISTS,
    DELETE_COLLECTION,
    DELETE_FILE,
)


async def get_all_collections_admin() -> list[dict[str, Any]]:
    """Returns every (app, collection, chunk_count) row across all apps."""
    rows = await get_pool().fetch(ALL_COLLECTIONS_ADMIN)
    return [
        {"app_name": r["app"], "collection_name": r["collection"], "chunk_count": r["chunk_count"]}
        for r in rows
    ]


async def get_vector_stats() -> tuple[int, int]:
    """Returns (total_collections, total_chunks) across all apps."""
    row = await get_pool().fetchrow(VECTOR_STATS)
    return int(row["cols"]), int(row["chunks"])


async def collection_exists(collection_name: str, app_name: str) -> bool:
    return bool(await get_pool().fetchval(COLLECTION_EXISTS, app_name, collection_name))


async def list_all_collections(app_name: str) -> list[str]:
    rows = await get_pool().fetch(LIST_COLLECTIONS, app_name)
    return [r["collection"] for r in rows]


async def list_files_in_collection(collection_name: str, app_name: str) -> list[str]:
    rows = await get_pool().fetch(LIST_FILES, app_name, collection_name)
    if not rows:
        raise ValueError(f"The collection '{collection_name}' does not exist.")
    return [r["source"] for r in rows]


async def delete_collection(collection_name: str, app_name: str) -> bool:
    result = await get_pool().execute(DELETE_COLLECTION, app_name, collection_name)
    return int(result.split()[-1]) > 0


async def delete_file_from_collection(collection_name: str, filename: str, app_name: str) -> bool:
    result = await get_pool().execute(DELETE_FILE, app_name, collection_name, filename)
    if int(result.split()[-1]) == 0:
        exists = await get_pool().fetchval(COLLECTION_EXISTS, app_name, collection_name)
        if not exists:
            raise ValueError(f"The collection '{collection_name}' does not exist.")
        raise ValueError(f"The file '{filename}' was not found in '{collection_name}'.")
    return True
