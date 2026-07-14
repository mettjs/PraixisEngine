import asyncio
from typing import Any

from src.config import HQ_ENABLED as _HQ_ENABLED
from src.utils.vectordb.chroma.client import get_owned_collection, get_questions_collection
from src.utils.vectordb.embeddings import embed
from src.utils.vectordb.fusion import (
    RAG_POOL_FACTOR,
    RAG_POOL_MIN,
    group_hits_by_source,
    merge_windows,
    rrf_fuse,
    source_filter,
)


def _ranked_keys(result: dict) -> list[tuple[str, int]]:
    """Chroma query result → ranked (source, chunk_index) keys, best first."""
    metas = result.get("metadatas")
    if not metas or not metas[0]:
        return []
    return [(str(m["source"]), int(m["chunk_index"])) for m in metas[0] if m]


def _chunk_search(collection, embedding: list[float], pool: int, source: str | None) -> list[tuple[str, int]]:
    """Dense search over the chunks collection. No FTS arm on this backend —
    pure vector is the degraded mode behind ``supports_hybrid=False``."""
    if source is not None:
        # Cap n_results against the matching subset: Chroma errors when asked
        # for more results than the filter can produce.
        filtered = collection.get(where={"source": source}, include=[]).get("ids") or []
        n = min(pool, len(filtered))
    else:
        n = min(pool, collection.count())
    if n == 0:
        return []
    return _ranked_keys(collection.query(
        query_embeddings=[embedding],
        n_results=n,
        where={"source": source} if source is not None else None,
        include=["metadatas"],
    ))


def _question_search(questions, embedding: list[float], pool: int, source: str | None) -> list[tuple[str, int]]:
    """Dense search over generated questions, collapsed to the parent chunk
    each one points to — mirrors pg's QUESTION_SEARCH dedup so one chunk
    matched by several of its questions never eats multiple result slots."""
    if source is not None:
        # Same guard as _chunk_search: n_results beyond the filter's matching
        # subset is a ChromaDB error, not a short result.
        filtered = questions.get(where={"source": source}, include=[]).get("ids") or []
        n = min(pool, len(filtered))
    else:
        n = min(pool, questions.count())
    if n == 0:
        return []
    result = questions.query(
        query_embeddings=[embedding],
        n_results=n,
        where={"source": source} if source is not None else None,
        include=["metadatas"],
    )
    metas = result.get("metadatas")
    if not metas or not metas[0]:
        return []
    ranked: list[tuple[str, int]] = []
    seen_parents: set[str] = set()
    for meta in metas[0]:
        if not meta:
            continue
        parent = str(meta["parent_chunk_id"])
        if parent in seen_parents:
            continue
        seen_parents.add(parent)
        ranked.append((str(meta["source"]), int(meta["chunk_index"])))
    return ranked


def _fetch_range(collection, source: str, lo: int, hi: int) -> str:
    rows = collection.get(
        where={"$and": [
            {"source": {"$eq": source}},
            {"chunk_index": {"$gte": lo}},
            {"chunk_index": {"$lte": hi}},
        ]},
        include=["documents", "metadatas"],
    )
    docs = rows.get("documents") or []
    metas = rows.get("metadatas") or []
    if len(metas) != len(docs):
        # Metadata unavailable or misaligned — fall back to Chroma's order
        # rather than risk dropping chunks via a truncated zip.
        return "\n\n".join(docs)
    ordered = sorted(zip(docs, metas), key=lambda pair: pair[1].get("chunk_index", 0) if pair[1] else 0)
    return "\n\n".join(doc for doc, _ in ordered)


async def query_rag_db(
    collection_name: str,
    app_name: str,
    question: str,
    n_results: int = 5,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    embedding = await asyncio.to_thread(embed, [question])
    src_filter = source_filter(metadata_filter)
    # Over-fetch from each search so fusion has a real candidate pool to work
    # with; the final n_results cut happens AFTER fusion, not inside either query.
    candidate_pool = max(n_results * RAG_POOL_FACTOR, RAG_POOL_MIN)

    def _run():
        collection = get_owned_collection(collection_name, app_name)

        ranked_lists = [_chunk_search(collection, embedding[0], candidate_pool, src_filter)]
        if _HQ_ENABLED:
            questions = get_questions_collection(collection_name, app_name)
            if questions is not None:
                ranked_lists.append(_question_search(questions, embedding[0], candidate_pool, src_filter))

        fused = rrf_fuse(ranked_lists, n_results)

        # Group hits by source (dict preserves fused-rank order), then merge
        # overlapping windows so duplicate content is never sent to the LLM.
        out: list[dict[str, str]] = []
        for source, indices in group_hits_by_source(fused).items():
            for lo, hi in merge_windows(indices):
                text = _fetch_range(collection, source, lo, hi)
                if text:
                    out.append({"source": source, "text": text})
        return out

    return await asyncio.to_thread(_run)


async def search_collection(
    collection_name: str,
    app_name: str,
    query: str,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    embedding = await asyncio.to_thread(embed, [query])

    def _run():
        try:
            collection = get_owned_collection(collection_name, app_name)
        except ValueError:
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        # Single dense query, no fusion — fetching beyond n_results buys nothing.
        n = min(n_results, collection.count())
        if n == 0:
            return []
        results = collection.query(
            query_embeddings=[embedding[0]],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        docs = results.get("documents")
        metas = results.get("metadatas")
        distances = results.get("distances")
        retrieved: list[dict[str, Any]] = []
        if docs and docs[0]:
            for i in range(len(docs[0])):
                meta = metas[0][i] if metas and metas[0] else {}
                distance = distances[0][i] if distances and distances[0] else None
                retrieved.append({
                    "source": str(meta.get("source", "Unknown")) if meta else "Unknown",
                    "text": docs[0][i],
                    "score": round(1 / (1 + distance), 3) if distance is not None else None,
                })
        return retrieved

    return await asyncio.to_thread(_run)


async def get_file_chunks(collection_name: str, app_name: str, filename: str) -> list[dict]:
    def _run():
        collection = get_owned_collection(collection_name, app_name)
        rows = collection.get(where={"source": filename}, include=["documents", "metadatas"])
        ids = rows.get("ids") or []
        if not ids:
            raise ValueError(f"No chunks found for document '{filename}' in this collection.")
        docs = rows.get("documents") or []
        metas = rows.get("metadatas") or []
        out = [
            {
                "id": str(cid),
                "chunk_index": int(meta.get("chunk_index", 0)) if meta else 0,
                "content": doc,
            }
            for cid, doc, meta in zip(ids, docs, metas)
        ]
        # Chroma does not guarantee retrieval order; return original sequence.
        out.sort(key=lambda r: r["chunk_index"])
        return out

    return await asyncio.to_thread(_run)


async def get_full_document_text(collection_name: str, app_name: str, filename: str) -> str:
    def _run():
        collection = get_owned_collection(collection_name, app_name)
        results = collection.get(where={"source": filename}, include=["documents", "metadatas"])
        documents = results.get("documents") if results else None
        if not documents:
            raise ValueError(f"No chunks found for document '{filename}' in this collection.")

        metas = results.get("metadatas")
        if not metas or len(metas) != len(documents):
            # Metadata unavailable or misaligned — fall back to Chroma's order
            # rather than risk dropping chunks via a truncated zip.
            return "\n\n".join(documents)

        # Chroma does not guarantee retrieval order; sort by the stored chunk_index
        # so the reconstructed document preserves its original sequence.
        ordered = sorted(
            zip(documents, metas),
            key=lambda pair: pair[1].get("chunk_index", float("inf")) if pair[1] else float("inf"),
        )
        return "\n\n".join(doc for doc, _ in ordered)

    return await asyncio.to_thread(_run)
