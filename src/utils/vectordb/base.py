"""Backend-neutral contract for vector stores.

Every retrieval backend (pgvector, Chroma) implements :class:`VectorStore`.
Controllers and shared modules talk only to this interface — obtained via
``src.utils.vectordb.get_vector_store()`` — never to a backend module directly.

Capability flags let shared code adapt without isinstance checks:

- ``supports_hybrid``: the backend fuses dense + full-text retrieval natively.
  Backends without it degrade to pure vector search; callers don't branch.
- ``supports_questions``: the backend can persist the hypothetical-question
  index. Question *generation* is backend-agnostic (see ``questions.py``);
  only storage and question-search are per-backend.
"""
from abc import ABC, abstractmethod
from typing import Any, ClassVar


class StaleChunksError(Exception):
    """Raised by ``store_questions`` when the parent chunks no longer exist.

    This happens when a file is deleted or re-uploaded while background
    question generation was still running. The newer upload regenerates its
    own questions, so callers treat this as "superseded", not as a failure.
    """


class VectorStore(ABC):
    supports_hybrid: ClassVar[bool]
    supports_questions: ClassVar[bool]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @abstractmethod
    async def init(self) -> None:
        """Connect / open storage and ensure the schema exists."""

    @abstractmethod
    async def close(self) -> None:
        """Release connections; safe to call when never initialized."""

    @abstractmethod
    async def ping(self) -> None:
        """Raises if the backend is unreachable."""

    # ── Collections ───────────────────────────────────────────────────────────

    @abstractmethod
    async def collection_exists(self, collection_name: str, app_name: str) -> bool: ...

    @abstractmethod
    async def list_collections(self, app_name: str) -> list[str]: ...

    @abstractmethod
    async def list_files(self, collection_name: str, app_name: str) -> list[str]:
        """Raises ValueError when the collection does not exist."""

    @abstractmethod
    async def delete_collection(self, collection_name: str, app_name: str) -> bool:
        """Returns False when there was nothing to delete."""

    @abstractmethod
    async def delete_file(self, collection_name: str, filename: str, app_name: str) -> bool:
        """Raises ValueError when the collection or file does not exist."""

    # ── Admin ─────────────────────────────────────────────────────────────────

    @abstractmethod
    async def all_collections_with_counts(self) -> list[dict[str, Any]]:
        """Every collection across all apps, as
        ``{"app_name", "collection_name", "chunk_count"}`` dicts."""

    @abstractmethod
    async def stats(self) -> tuple[int, int]:
        """Returns (total_collections, total_chunks) across all apps."""

    # ── Ingestion ─────────────────────────────────────────────────────────────

    @abstractmethod
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
        """Chunk, embed, and store a document, replacing any prior copy.

        Returns the inserted chunk rows as dicts with keys ``id``,
        ``chunk_index``, and ``content`` so the caller can drive deferred
        hypothetical-question generation against the parent chunks.
        """

    @abstractmethod
    async def store_questions(
        self,
        app_name: str,
        collection_name: str,
        source: str,
        entries: list[tuple[dict, str, list[float]]],
    ) -> None:
        """Persist generated questions for a file's chunks.

        ``entries`` is a list of ``(parent_chunk, question_text, embedding)``
        where ``parent_chunk`` is a row dict returned by :meth:`add_file`.
        Raises :class:`StaleChunksError` when the parent chunks are gone.
        """

    # ── Retrieval ─────────────────────────────────────────────────────────────

    @abstractmethod
    async def query(
        self,
        collection_name: str,
        app_name: str,
        question: str,
        n_results: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        """RAG retrieval: returns ``{"source", "text"}`` context blocks,
        window-expanded and fused across whatever search paths the backend
        supports (dense, full-text, question index)."""

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        app_name: str,
        query: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Admin search: raw ranked chunks as ``{"source", "text", "score"}``.
        Raises ValueError when the collection does not exist."""

    @abstractmethod
    async def full_document(self, collection_name: str, app_name: str, filename: str) -> str:
        """The document's chunks re-joined in order.
        Raises ValueError when no chunks exist for the file."""
