import asyncio
import threading
import uuid

from src.utils.vectordb.base import StaleChunksError
from src.utils.vectordb.chroma.client import (
    drop_collection,
    ensure_name_fits,
    get_client,
    get_owned_collection,
    get_questions_collection,
    is_owned,
    questions_name,
    scoped_name,
)
from src.utils.vectordb.chunking import character_chunk, semantic_chunk
from src.utils.vectordb.embeddings import embed

# One lock per collection, guarding the "was it empty → insert → roll back"
# window below. Ingestion runs in a worker thread, so two uploads racing into
# the same brand-new collection would otherwise both read count() == 0 and the
# loser's rollback would delete the winner's chunks. Chroma has no
# transactions, so this is the only way to make that decision atomic.
#
# The dict grows one small entry per collection ever written to in this
# process, which is bounded by the deployment's collection count.
_collection_locks: dict[str, threading.Lock] = {}
_collection_locks_guard = threading.Lock()


def _collection_lock(scoped: str) -> threading.Lock:
    with _collection_locks_guard:
        return _collection_locks.setdefault(scoped, threading.Lock())


async def add_file_to_rag_db(
    text: str,
    collection_name: str,
    filename: str,
    app_name: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 150,
    chunking_strategy: str = "semantic",
) -> list[dict]:
    """Chunk, embed, and store a document, replacing any prior copy of the file.

    Returns the inserted chunk rows as dicts with keys ``id``, ``chunk_index``,
    and ``content`` so the caller can drive deferred hypothetical-question
    generation against the parent chunks.
    """
    if chunking_strategy == "semantic":
        chunks = await asyncio.to_thread(semantic_chunk, text, max_chunk_chars=chunk_size)
    else:
        chunks = await asyncio.to_thread(character_chunk, text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not chunks:
        raise ValueError("Document produced no chunks; nothing to ingest.")
    embeddings = await asyncio.to_thread(embed, chunks)

    def _run():
        ensure_name_fits(app_name, collection_name)
        scoped = scoped_name(app_name, collection_name)
        collection = get_client().get_or_create_collection(
            name=scoped,
            metadata={"app": app_name},
        )
        if not is_owned(collection, app_name):
            raise ValueError("Access denied: You do not own this collection.")

        existing = collection.get(where={"source": filename}, include=[])
        old_ids: list[str] = existing.get("ids") or []

        ids = [uuid.uuid4().hex for _ in chunks]
        metadatas = [
            {"source": filename, "app": app_name, "chunk_index": i}
            for i in range(len(chunks))
        ]

        # Add new chunks first — if this fails, old data is still intact.
        # get_or_create above may have just created this collection, though, and
        # a failed insert would strand it empty. That breaks the invariant the
        # deletion path maintains — a collection exists exactly as long as it
        # holds chunks — which pgvector gets for free (no rows inserted, no
        # collection). Ownership was verified above, so rolling an empty
        # collection back can only drop something this app owns and that no one
        # can retrieve from. A replace (old_ids non-empty) leaves count > 0, so
        # an existing collection is never dropped here.
        #
        # The emptiness check, the insert and the rollback are held under one
        # per-collection lock, and the rollback re-checks emptiness rather than
        # trusting the reading taken before the insert: a concurrent upload into
        # the same new collection may have succeeded in between, and dropping
        # the collection would destroy its chunks after it was told they were
        # stored. The lock covers this process; the re-check is what keeps a
        # second worker or replica from doing the damage.
        with _collection_lock(scoped):
            was_empty = collection.count() == 0
            try:
                collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
            except Exception:
                if was_empty and collection.count() == 0:
                    drop_collection(collection_name, app_name)
                raise
        if old_ids:
            collection.delete(ids=old_ids)

        # Purge the replaced file's question index; the new upload regenerates
        # it (when improved_search is on). Without an FK to cascade for us,
        # this is the explicit equivalent of pg's ON DELETE CASCADE.
        questions = get_questions_collection(collection_name, app_name)
        if questions is not None:
            questions.delete(where={"source": filename})

        return [{"id": cid, "chunk_index": i, "content": chunk} for i, (cid, chunk) in enumerate(zip(ids, chunks))]

    return await asyncio.to_thread(_run)


async def store_questions(
    app_name: str,
    collection_name: str,
    source: str,
    entries: list[tuple[dict, str, list[float]]],
) -> None:
    """Persist generated questions in the parallel ``__hyq`` collection.

    Chroma has no foreign keys, so staleness (the file was deleted or
    re-uploaded mid-generation) is detected by checking that the parent chunk
    ids still exist immediately before writing. A write that lands in the
    narrow window after that check can leave orphaned question rows; they are
    harmless — their (source, chunk_index) keys resolve to nothing at query
    time — and are purged on the file's next upload or deletion.
    """
    def _run():
        collection = get_owned_collection(collection_name, app_name)

        parent_ids = list({chunk["id"] for chunk, _, _ in entries})
        surviving = set(collection.get(ids=parent_ids, include=[]).get("ids") or [])
        if not surviving:
            raise StaleChunksError(f"Parent chunks for '{source}' no longer exist.")
        kept = [e for e in entries if e[0]["id"] in surviving]

        questions = get_client().get_or_create_collection(
            name=questions_name(app_name, collection_name),
            metadata={"app": app_name, "hyq": True},
        )
        questions.add(
            ids=[uuid.uuid4().hex for _ in kept],
            documents=[q for _, q, _ in kept],
            embeddings=[emb for _, _, emb in kept],
            metadatas=[
                {
                    "app": app_name,
                    "source": source,
                    "parent_chunk_id": chunk["id"],
                    "chunk_index": chunk["chunk_index"],
                }
                for chunk, _, _ in kept
            ],
        )

    await asyncio.to_thread(_run)
