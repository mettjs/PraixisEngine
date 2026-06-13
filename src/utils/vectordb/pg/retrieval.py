import asyncio
import re
from typing import Any

from src.config import HQ_ENABLED as _HQ_ENABLED
from src.utils.vectordb.pg.pool import get_pool
from src.utils.vectordb.embeddings import embed
from src.utils.vectordb.fusion import (
    ADMIN_POOL_FACTOR,
    ADMIN_POOL_MIN,
    RAG_POOL_FACTOR,
    RAG_POOL_MIN,
    group_hits_by_source,
    merge_windows,
    rrf_fuse,
    source_filter,
)
from src.utils.vectordb.pg.constants import (
    COLLECTION_EXISTS,
    FULL_DOCUMENT,
    HYBRID_SEARCH,
    QUESTION_SEARCH,
    WINDOW_CHUNKS,
)

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


async def _fetch_range(app: str, collection: str, source: str, lo: int, hi: int) -> str:
    rows = await get_pool().fetch(WINDOW_CHUNKS, app, collection, source, lo, hi)
    return "\n\n".join(r["content"] for r in rows)


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

    # Hybrid (dense + sparse) over the source text, and — when enabled — a dense
    # search over generated questions, run concurrently. Both return ranked
    # (source, chunk_index) candidates on the same key so they fuse directly.
    searches = [
        get_pool().fetch(
            HYBRID_SEARCH,
            embedding[0], app_name, collection_name,
            candidate_pool, _fts_query(question),
            src_filter, candidate_pool,
        )
    ]
    if _HQ_ENABLED:
        searches.append(
            get_pool().fetch(
                QUESTION_SEARCH,
                embedding[0], app_name, collection_name,
                candidate_pool, src_filter, candidate_pool,
            )
        )

    results = await asyncio.gather(*searches)
    ranked_lists = [
        [(r["source"], r["chunk_index"]) for r in rows] for rows in results
    ]
    fused = rrf_fuse(ranked_lists, n_results)

    # Group hits by source (dict preserves fused-rank order), then merge
    # overlapping windows so duplicate content is never sent to the LLM.
    fetch_tasks: list = []
    result_sources: list[str] = []
    for source, indices in group_hits_by_source(fused).items():
        for lo, hi in merge_windows(indices):
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
        max(n_results * ADMIN_POOL_FACTOR, ADMIN_POOL_MIN), _fts_query(query), None, n_results,
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
