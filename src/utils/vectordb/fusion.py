"""Backend-agnostic retrieval plumbing: rank fusion and window expansion.

Both backends retrieve ranked ``(source, chunk_index)`` candidate lists from
whatever search paths they support, then share this module to fuse the lists
and merge neighbor windows. Only the fetches themselves are backend-specific.
"""
from typing import Any

_RRF_K = 60  # rank-fusion damping; matches the constant used inside pg's HYBRID_SEARCH

# Candidate over-fetch before rank fusion. Fusion only works if each ranked
# list is wider than the final cut, and the RAG path fuses up to three lists
# (dense, sparse, question index) — so it over-fetches more aggressively than
# the single-query admin search. The asymmetry is deliberate; don't "fix" one
# side to match the other.
RAG_POOL_FACTOR = 4
RAG_POOL_MIN = 40   # sized to exploit the HNSW ef_search ceiling (60)
ADMIN_POOL_FACTOR = 3
ADMIN_POOL_MIN = 15

CONTEXT_WINDOW = 1  # neighbor chunks to include on each side of every hit


def source_filter(metadata_filter: dict[str, Any] | None) -> str | None:
    """The one metadata filter both backends honor: restrict to a source file."""
    if metadata_filter and isinstance(metadata_filter.get("source"), str):
        return metadata_filter["source"]
    return None


def rrf_fuse(
    ranked_lists: list[list[tuple[str, int]]], limit: int
) -> list[tuple[str, int]]:
    """Reciprocal-rank-fuse several ordered (source, chunk_index) lists.

    Each list is already ranked best-first. A key's score is the sum of
    1/(K + rank) across the lists it appears in, so a chunk surfaced by two
    search paths outranks one found by only one path. Returns the top
    ``limit`` keys.
    """
    scores: dict[tuple[str, int], float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
    return sorted(scores, key=lambda k: scores[k], reverse=True)[:limit]


def merge_windows(chunk_indices: list[int]) -> list[tuple[int, int]]:
    """Merge window ranges for hits from the same source.

    When two retrieved chunks are close enough that their expanded windows
    overlap, joining them into one contiguous range avoids sending the shared
    text to the LLM twice and produces a more readable context block.
    """
    sorted_idx = sorted(set(chunk_indices))
    lo = max(0, sorted_idx[0] - CONTEXT_WINDOW)
    hi = sorted_idx[0] + CONTEXT_WINDOW
    merged: list[tuple[int, int]] = []
    for idx in sorted_idx[1:]:
        new_lo = max(0, idx - CONTEXT_WINDOW)
        new_hi = idx + CONTEXT_WINDOW
        if new_lo <= hi + 1:
            hi = max(hi, new_hi)
        else:
            merged.append((lo, hi))
            lo, hi = new_lo, new_hi
    merged.append((lo, hi))
    return merged


def group_hits_by_source(fused: list[tuple[str, int]]) -> dict[str, list[int]]:
    """Group fused (source, chunk_index) hits by source, preserving rank order."""
    source_hits: dict[str, list[int]] = {}
    for source, chunk_index in fused:
        source_hits.setdefault(source, []).append(chunk_index)
    return source_hits
