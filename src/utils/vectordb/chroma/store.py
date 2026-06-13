"""Chroma implementation of the VectorStore contract.

Embedded/persistent ChromaDB — no extra server beyond the API itself. Vectors
come precomputed from the shared embeddings module (same model as pgvector),
so the two backends are interchangeable per deployment. Retrieval is pure
vector (no full-text arm); the hypothetical-question index lives in a parallel
``__hyq`` collection and fuses into queries exactly like pg's question table.
"""
import asyncio
from typing import Any

from src.utils.vectordb.base import VectorStore
from src.utils.vectordb.chroma import collections, ingestion, retrieval
from src.utils.vectordb.chroma.client import close_client, get_client, init_client


class ChromaStore(VectorStore):
    supports_hybrid = False      # no FTS — degrades to pure vector search
    supports_questions = True    # parallel {collection}__hyq collection

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def init(self) -> None:
        await asyncio.to_thread(init_client)

    async def close(self) -> None:
        close_client()

    async def ping(self) -> None:
        await asyncio.to_thread(lambda: get_client().heartbeat())

    # ── Collections ───────────────────────────────────────────────────────────

    async def collection_exists(self, collection_name: str, app_name: str) -> bool:
        return await collections.collection_exists(collection_name, app_name)

    async def list_collections(self, app_name: str) -> list[str]:
        return await collections.list_all_collections(app_name)

    async def list_files(self, collection_name: str, app_name: str) -> list[str]:
        return await collections.list_files_in_collection(collection_name, app_name)

    async def delete_collection(self, collection_name: str, app_name: str) -> bool:
        return await collections.delete_collection(collection_name, app_name)

    async def delete_file(self, collection_name: str, filename: str, app_name: str) -> bool:
        return await collections.delete_file_from_collection(collection_name, filename, app_name)

    # ── Admin ─────────────────────────────────────────────────────────────────

    async def all_collections_with_counts(self) -> list[dict[str, Any]]:
        return await collections.get_all_collections_admin()

    async def stats(self) -> tuple[int, int]:
        return await collections.get_vector_stats()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    async def add_file(
        self,
        text: str,
        collection_name: str,
        filename: str,
        app_name: str,
        chunk_size: int = 2000,
        chunk_overlap: int = 150,
        chunking_strategy: str = "semantic",
    ) -> list[dict]:
        return await ingestion.add_file_to_rag_db(
            text, collection_name, filename, app_name,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            chunking_strategy=chunking_strategy,
        )

    async def store_questions(
        self,
        app_name: str,
        collection_name: str,
        source: str,
        entries: list[tuple[dict, str, list[float]]],
    ) -> None:
        await ingestion.store_questions(app_name, collection_name, source, entries)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    async def query(
        self,
        collection_name: str,
        app_name: str,
        question: str,
        n_results: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        return await retrieval.query_rag_db(
            collection_name, app_name, question,
            n_results=n_results, metadata_filter=metadata_filter,
        )

    async def search(
        self,
        collection_name: str,
        app_name: str,
        query: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        return await retrieval.search_collection(collection_name, app_name, query, n_results=n_results)

    async def full_document(self, collection_name: str, app_name: str, filename: str) -> str:
        return await retrieval.get_full_document_text(collection_name, app_name, filename)
