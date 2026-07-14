"""PostgreSQL + pgvector implementation of the VectorStore contract.

Thin class facade over the focused modules in this package (pool, collections,
ingestion, retrieval); the SQL lives in ``constants.py``.
"""
import uuid
from typing import Any

import asyncpg

from src.utils.vectordb.base import StaleChunksError, VectorStore
from src.utils.vectordb.pg import collections, ingestion, pool, retrieval
from src.utils.vectordb.pg.constants import COUNT_CHUNKS, COUNT_QUESTIONS, DELETE_QUESTIONS, INSERT_QUESTION


class PgVectorStore(VectorStore):
    supports_hybrid = True       # dense + FTS fused in SQL (HYBRID_SEARCH)
    supports_questions = True    # chunk_questions table, FK ON DELETE CASCADE

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def init(self) -> None:
        await pool.init_db()

    async def close(self) -> None:
        await pool.close_db()

    async def ping(self) -> None:
        await pool.ping()

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
        rows = [
            (
                uuid.uuid4().hex,
                app_name,
                collection_name,
                source,
                chunk["id"],
                chunk["chunk_index"],
                question,
                emb,
            )
            for chunk, question, emb in entries
        ]
        try:
            await pool.get_pool().executemany(INSERT_QUESTION, rows)
        except asyncpg.ForeignKeyViolationError as e:
            # The parent chunks were deleted or replaced while questions were
            # being generated; the FK keeps the index consistent for us.
            raise StaleChunksError(str(e)) from e

    async def delete_questions(self, app_name: str, collection_name: str, source: str) -> None:
        await pool.get_pool().execute(DELETE_QUESTIONS, app_name, collection_name, source)

    async def count_questions(self, app_name: str, collection_name: str, source: str) -> int:
        return int(await pool.get_pool().fetchval(COUNT_QUESTIONS, app_name, collection_name, source))

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

    async def file_chunks(self, collection_name: str, app_name: str, filename: str) -> list[dict]:
        return await retrieval.get_file_chunks(collection_name, app_name, filename)

    async def count_chunks(self, collection_name: str, app_name: str, filename: str) -> int:
        return int(await pool.get_pool().fetchval(COUNT_CHUNKS, app_name, collection_name, filename))
