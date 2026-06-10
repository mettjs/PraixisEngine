import asyncio
import re
from typing import Any

from src.config import HQ_ENABLED as _HQ_ENABLED
from src.utils.vectordb.pool import get_pool
from src.utils.vectordb.embeddings import embed
from src.utils.vectordb.constants import (
    COLLECTION_EXISTS,
    FULL_DOCUMENT,
    HYBRID_SEARCH,
    QUESTION_SEARCH,
    WINDOW_CHUNKS,
)


def _source_filter(metadata_filter: dict[str, Any] | None) -> str | None:
    if metadata_filter and isinstance(metadata_filter.get("source"), str):
        return metadata_filter["source"]
    return None


_WORD_NUM_RE = re.compile(r"\b(\w+)\s+(\d+)\b")


def _fts_query(text: str) -> str:
    """Build an FTS string for websearch_to_tsquery.

    Single-digit numbers have near-zero IDF in most documents (page numbers,
    list items, dates), so OR semantics alone can't distinguish 'articulo 5'
    from a chunk that just happens to contain a '5'.  Any word+number pair in
    the query is promoted to a phrase match ('articulo' <-> '5'), which requires
    the tokens to be adjacent in the tsvector — exactly how headings are stored.
    This is language-agnostic: it works for 'article 5', 'section 3',
    'Artikel 5', 'paragraphe 2', etc.  OR terms stay alongside as fallback.
    """
    or_terms = " OR ".join(re.findall(r"\w+", text))
    phrases = [f'"{w} {n}"' for w, n in _WORD_NUM_RE.findall(text)]
    if phrases:
        phrase_part = " OR ".join(phrases)
        return f"{phrase_part} OR {or_terms}" if or_terms else phrase_part
    return or_terms if or_terms else text


_CONTEXT_WINDOW = 1  # neighbor chunks to include on each side of every hit


def _merge_windows(chunk_indices: list[int]) -> list[tuple[int, int]]:
    """Merge window ranges for hits from the same source.

    When two retrieved chunks are close enough that their expanded windows
    overlap, joining them into one contiguous range avoids sending the shared
    text to the LLM twice and produces a more readable context block.
    """
    sorted_idx = sorted(set(chunk_indices))
    lo = max(0, sorted_idx[0] - _CONTEXT_WINDOW)
    hi = sorted_idx[0] + _CONTEXT_WINDOW
    merged: list[tuple[int, int]] = []
    for idx in sorted_idx[1:]:
        new_lo = max(0, idx - _CONTEXT_WINDOW)
        new_hi = idx + _CONTEXT_WINDOW
        if new_lo <= hi + 1:
            hi = max(hi, new_hi)
        else:
            merged.append((lo, hi))
            lo, hi = new_lo, new_hi
    merged.append((lo, hi))
    return merged


async def _fetch_range(app: str, collection: str, source: str, lo: int, hi: int) -> str:
    rows = await get_pool().fetch(WINDOW_CHUNKS, app, collection, source, lo, hi)
    return "\n\n".join(r["content"] for r in rows)


_RRF_K = 60  # rank-fusion damping; matches the constant used inside HYBRID_SEARCH

# Candidate over-fetch before rank fusion. Fusion only works if each ranked
# list is wider than the final cut, and the RAG path fuses up to three lists
# (dense, sparse, question index) — so it over-fetches more aggressively than
# the single-query admin search. The asymmetry is deliberate; don't "fix" one
# side to match the other.
_RAG_POOL_FACTOR = 4
_RAG_POOL_MIN = 40   # sized to exploit the HNSW ef_search ceiling (60)
_ADMIN_POOL_FACTOR = 3
_ADMIN_POOL_MIN = 15


def _rrf_fuse(
    ranked_lists: list[list[tuple[str, int]]], limit: int
) -> list[tuple[str, int]]:
    """Reciprocal-rank-fuse several ordered (source, chunk_index) lists.

    Each list is already ranked best-first. A key's score is the sum of
    1/(K + rank) across the lists it appears in, so a chunk surfaced by both the
    hybrid search and the question search outranks one found by only one path.
    Returns the top ``limit`` keys.
    """
    scores: dict[tuple[str, int], float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
    return sorted(scores, key=lambda k: scores[k], reverse=True)[:limit]


async def query_rag_db(
    collection_name: str,
    app_name: str,
    question: str,
    n_results: int = 5,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    embedding = await asyncio.to_thread(embed, [question])
    source_filter = _source_filter(metadata_filter)
    # Over-fetch from each search so fusion has a real candidate pool to work
    # with; the final n_results cut happens AFTER fusion, not inside either query.
    candidate_pool = max(n_results * _RAG_POOL_FACTOR, _RAG_POOL_MIN)

    # Hybrid (dense + sparse) over the source text, and — when enabled — a dense
    # search over generated questions, run concurrently. Both return ranked
    # (source, chunk_index) candidates on the same key so they fuse directly.
    searches = [
        get_pool().fetch(
            HYBRID_SEARCH,
            embedding[0], app_name, collection_name,
            candidate_pool, _fts_query(question),
            source_filter, candidate_pool,
        )
    ]
    if _HQ_ENABLED:
        searches.append(
            get_pool().fetch(
                QUESTION_SEARCH,
                embedding[0], app_name, collection_name,
                candidate_pool, source_filter, candidate_pool,
            )
        )

    results = await asyncio.gather(*searches)
    ranked_lists = [
        [(r["source"], r["chunk_index"]) for r in rows] for rows in results
    ]
    fused = _rrf_fuse(ranked_lists, n_results)

    # Group hits by source (dict preserves fused-rank order), then merge
    # overlapping windows so duplicate content is never sent to the LLM.
    source_hits: dict[str, list[int]] = {}
    for source, chunk_index in fused:
        source_hits.setdefault(source, []).append(chunk_index)

    fetch_tasks: list = []
    result_sources: list[str] = []
    for source, indices in source_hits.items():
        for lo, hi in _merge_windows(indices):
            fetch_tasks.append(_fetch_range(app_name, collection_name, source, lo, hi))
            result_sources.append(source)

    texts = await asyncio.gather(*fetch_tasks)
    return [{"source": src, "text": text} for src, text in zip(result_sources, texts)]


async def search_collection(
    collection_name: str,
    app_name: str,
    query: str,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    exists = await get_pool().fetchval(COLLECTION_EXISTS, app_name, collection_name)
    if not exists:
        raise ValueError(f"Collection '{collection_name}' does not exist.")

    embedding = await asyncio.to_thread(embed, [query])
    rows = await get_pool().fetch(
        HYBRID_SEARCH,
        embedding[0], app_name, collection_name,
        max(n_results * _ADMIN_POOL_FACTOR, _ADMIN_POOL_MIN), _fts_query(query), None, n_results,
    )
    return [
        {"source": r["source"], "text": r["content"], "score": round(float(r["rrf_score"]), 4)}
        for r in rows
    ]


async def get_full_document_text(collection_name: str, app_name: str, filename: str) -> str:
    rows = await get_pool().fetch(FULL_DOCUMENT, app_name, collection_name, filename)
    if not rows:
        raise ValueError(f"No chunks found for document '{filename}' in this collection.")
    return "\n\n".join(r["content"] for r in rows)
