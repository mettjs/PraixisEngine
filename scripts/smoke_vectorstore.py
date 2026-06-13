"""Functional smoke test for the VectorStore backends.

Exercises the full lifecycle through the factory: ingest, query (with and
without the question index), search, full document, deletes. Backend comes
from VECTOR_BACKEND; for chroma point CHROMA_PATH at a scratch directory.
Global stats assertions are delta-based so a non-empty pg database passes.

Run: PYTHONPATH=. VECTOR_BACKEND=chroma|pgvector uv run python scripts/smoke_vectorstore.py
"""
import asyncio
import uuid

from src.utils.vectordb import get_vector_store
from src.utils.vectordb.base import StaleChunksError
from src.utils.vectordb.embeddings import embed

APP = f"smoke{uuid.uuid4().hex[:8]}"
COL = "docs"

DOC = "\n\n".join(
    f"Articulo {i}.- Disposicion numero {i}. " + ("Texto de relleno legal. " * 10)
    for i in range(1, 9)
)


def _our_entry(cols: list[dict]) -> dict | None:
    matches = [c for c in cols if c["app_name"] == APP and c["collection_name"] == COL]
    return matches[0] if matches else None


async def main() -> None:
    store = get_vector_store()
    print("backend:", type(store).__name__)
    await store.init()
    await store.ping()

    # Ingest two files; character chunking to force multiple chunks.
    rows = await store.add_file(DOC, COL, "ley.txt", APP, chunk_size=300, chunk_overlap=0, chunking_strategy="character")
    await store.add_file("Documento corto sobre garantias y depositos bancarios.", COL, "otro.txt", APP)
    assert len(rows) > 3, f"expected several chunks, got {len(rows)}"
    assert all(set(r) == {"id", "chunk_index", "content"} for r in rows)
    print(f"ingested ley.txt as {len(rows)} chunks")

    # Re-upload must replace, not duplicate.
    rows2 = await store.add_file(DOC, COL, "ley.txt", APP, chunk_size=300, chunk_overlap=0, chunking_strategy="character")
    entry = _our_entry(await store.all_collections_with_counts())
    assert entry and entry["chunk_count"] == len(rows2) + 1, entry
    print("re-upload replaced prior copy:", entry)

    assert await store.collection_exists(COL, APP)
    assert not await store.collection_exists("nope", APP)
    assert sorted(await store.list_files(COL, APP)) == ["ley.txt", "otro.txt"]
    assert await store.list_collections(APP) == [COL]
    n_cols, n_chunks = await store.stats()
    assert n_cols >= 1 and n_chunks >= len(rows2) + 1

    # Query without question index.
    hits = await store.query(COL, APP, "disposicion numero 5", n_results=3)
    assert hits and all(set(h) == {"source", "text"} for h in hits), hits
    print(f"query (no HQ): {len(hits)} context blocks")

    # Query with source filter.
    hits = await store.query(COL, APP, "garantias", n_results=2, metadata_filter={"source": "otro.txt"})
    assert hits and all(h["source"] == "otro.txt" for h in hits), hits
    print("source-filtered query OK")

    # Store questions (simulating the generation pass) and query again.
    entries = [
        (rows2[2], "que dice la disposicion numero tres?", embed(["que dice la disposicion numero tres?"])[0]),
        (rows2[4], "donde habla del numero cinco?", embed(["donde habla del numero cinco?"])[0]),
    ]
    await store.store_questions(APP, COL, "ley.txt", entries)
    hits = await store.query(COL, APP, "que dice la disposicion numero tres?", n_results=2)
    assert hits, "fused query returned nothing"
    print(f"query (HQ fused): {len(hits)} context blocks, top source={hits[0]['source']}")

    # Source-filtered query WITH a populated question index: the candidate pool
    # (40) far exceeds the filter's matching questions, which must be clamped,
    # not passed through as n_results.
    hits = await store.query(COL, APP, "que dice la disposicion numero tres?", n_results=2, metadata_filter={"source": "ley.txt"})
    assert hits and all(h["source"] == "ley.txt" for h in hits), hits
    print("source-filtered query with question index OK")

    # The question index must not leak into listings.
    assert await store.list_collections(APP) == [COL]

    # Stale parents are detected.
    try:
        await store.store_questions(APP, COL, "ley.txt", [({"id": "deadbeef", "chunk_index": 0}, "q?", embed(["q?"])[0])])
        raise AssertionError("expected StaleChunksError")
    except StaleChunksError:
        print("stale-parent detection OK")

    # Admin search.
    results = await store.search(COL, APP, "disposicion numero 2", n_results=3)
    assert results and all({"source", "text", "score"} <= set(r) for r in results), results
    assert len(results) <= 3
    print(f"admin search: {len(results)} results, top score={results[0]['score']}")

    # Full document reconstruction preserves order.
    full = await store.full_document(COL, APP, "ley.txt")
    assert full.index("Articulo 1") < full.index("Articulo 8")
    print("full document reconstruction OK")

    # delete_file purges the file's questions too.
    assert await store.delete_file(COL, "ley.txt", APP)
    hits = await store.query(COL, APP, "que dice la disposicion numero tres?", n_results=2)
    assert all(h["source"] == "otro.txt" for h in hits), hits
    try:
        await store.delete_file(COL, "ley.txt", APP)
        raise AssertionError("expected ValueError for missing file")
    except ValueError:
        pass
    print("delete_file + question purge OK")

    # delete_collection removes the question index as well.
    assert await store.delete_collection(COL, APP)
    assert not await store.delete_collection(COL, APP)
    assert not await store.collection_exists(COL, APP)
    assert _our_entry(await store.all_collections_with_counts()) is None
    print("delete_collection OK")

    await store.close()
    print(f"{type(store).__name__} SMOKE OK")


if __name__ == "__main__":
    asyncio.run(main())
