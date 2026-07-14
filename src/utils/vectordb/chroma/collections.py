import asyncio
from typing import Any, Set

from src.utils.vectordb.chroma.client import (
    get_client,
    get_owned_collection,
    get_questions_collection,
    questions_name,
    scoped_name,
)


def _is_questions_index(metadata: dict | None) -> bool:
    return bool(metadata) and bool(metadata.get("hyq"))


async def get_all_collections_admin() -> list[dict[str, Any]]:
    """Returns every (app, collection, chunk_count) row across all apps."""
    def _run():
        out: list[dict[str, Any]] = []
        for col in get_client().list_collections():
            if not col.metadata or "app" not in col.metadata or _is_questions_index(col.metadata):
                continue
            app = str(col.metadata["app"])
            prefix = f"{app}."
            name = col.name[len(prefix):] if col.name.startswith(prefix) else col.name
            out.append({
                "app_name": app,
                "collection_name": name,
                "chunk_count": col.count(),
            })
        return sorted(out, key=lambda c: (c["app_name"], c["collection_name"]))

    return await asyncio.to_thread(_run)


async def get_vector_stats() -> tuple[int, int]:
    """Returns (total_collections, total_chunks) across all apps."""
    collections = await get_all_collections_admin()
    return len(collections), sum(c["chunk_count"] for c in collections)


async def collection_exists(collection_name: str, app_name: str) -> bool:
    def _run():
        try:
            get_owned_collection(collection_name, app_name)
            return True
        except ValueError:
            return False

    return await asyncio.to_thread(_run)


async def list_all_collections(app_name: str) -> list[str]:
    def _run():
        prefix = f"{app_name}."
        return [
            col.name[len(prefix):]
            for col in get_client().list_collections()
            if col.metadata
            and col.metadata.get("app") == app_name
            and not _is_questions_index(col.metadata)
        ]

    return await asyncio.to_thread(_run)


async def list_files_in_collection(collection_name: str, app_name: str) -> list[str]:
    def _run():
        collection = get_owned_collection(collection_name, app_name)
        results = collection.get(include=["metadatas"])
        unique_files: Set[str] = set()
        for meta in results.get("metadatas") or []:
            if meta and "source" in meta:
                unique_files.add(str(meta["source"]))
        return list(unique_files)

    return await asyncio.to_thread(_run)


async def delete_collection(collection_name: str, app_name: str) -> bool:
    def _run():
        try:
            get_owned_collection(collection_name, app_name)
        except ValueError:
            return False
        get_client().delete_collection(name=scoped_name(app_name, collection_name))
        # The question index shares the collection's lifecycle.
        if get_questions_collection(collection_name, app_name) is not None:
            get_client().delete_collection(name=questions_name(app_name, collection_name))
        return True

    return await asyncio.to_thread(_run)


async def count_chunks_for_file(collection_name: str, app_name: str, source: str) -> int:
    def _run():
        collection = get_owned_collection(collection_name, app_name)
        return len(collection.get(where={"source": source}, include=[]).get("ids") or [])

    return await asyncio.to_thread(_run)


async def delete_questions_for_file(collection_name: str, app_name: str, source: str) -> None:
    def _run():
        questions = get_questions_collection(collection_name, app_name)
        if questions is not None:
            questions.delete(where={"source": source})

    await asyncio.to_thread(_run)


async def count_questions_for_file(collection_name: str, app_name: str, source: str) -> int:
    def _run():
        questions = get_questions_collection(collection_name, app_name)
        if questions is None:
            return 0
        return len(questions.get(where={"source": source}, include=[]).get("ids") or [])

    return await asyncio.to_thread(_run)


async def delete_file_from_collection(collection_name: str, filename: str, app_name: str) -> bool:
    def _run():
        collection = get_owned_collection(collection_name, app_name)
        existing = collection.get(where={"source": filename}, include=[])
        if not existing or not existing.get("ids"):
            raise ValueError(f"The file '{filename}' was not found in '{collection_name}'.")
        collection.delete(where={"source": filename})
        questions = get_questions_collection(collection_name, app_name)
        if questions is not None:
            questions.delete(where={"source": filename})
        return True

    return await asyncio.to_thread(_run)
