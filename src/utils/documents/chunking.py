"""Text chunking strategies: semantic (embedding-based) and character (recursive fixed-size)."""
import re
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.documents.embeddings import embed

_SENTENCE_RE = re.compile(r'(?<=[.!?…])\s+|\n{2,}')
_MIN_SENTENCE_CHARS = 25
_FALLBACK_SEPARATORS = ["\n\n", "\n", r"(?<=\. )", " ", ""]


def _split_sentences(text: str) -> list[str]:
    raw = _SENTENCE_RE.split(text.strip())
    parts = [p.strip() for p in raw if p.strip()]
    if not parts:
        return []
    # Merge short fragments into the preceding sentence so abbreviations like
    # "Art.", "núm.", "párr.", "Sr." don't become isolated embedding tokens.
    merged = [parts[0]]
    for part in parts[1:]:
        if len(part) < _MIN_SENTENCE_CHARS:
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)
    return merged


def semantic_chunk(
    text: str,
    breakpoint_percentile: int = 95,
    min_chunk_chars: int = 200,
    max_chunk_chars: int = 2000,
) -> list[str]:
    """Split text at natural semantic boundaries using embedding similarity.

    Sentences are embedded in a sliding window; wherever consecutive windows
    drop below a similarity threshold, a new chunk begins. Oversized chunks
    are split further with RecursiveCharacterTextSplitter as a safety net.

    breakpoint_percentile: higher = fewer, larger chunks. 95 means only the
    sharpest 5% of similarity drops trigger a cut.
    """
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return [text.strip()] if text.strip() else []

    # Each window = sentence ± 1 neighbor for local context
    windows = [
        " ".join(sentences[max(0, i - 1):min(len(sentences), i + 2)])
        for i in range(len(sentences))
    ]
    emb = np.asarray(embed(windows), dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_norm = emb / np.maximum(norms, 1e-10)
    similarities = (emb_norm[:-1] * emb_norm[1:]).sum(axis=1).tolist()

    threshold = float(np.percentile(similarities, 100 - breakpoint_percentile))
    breakpoints = {i + 1 for i, sim in enumerate(similarities) if sim <= threshold}

    # Accumulate sentences into chunks; skip a breakpoint when the current
    # chunk is still below min_chunk_chars to avoid tiny orphan chunks.
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for i, sentence in enumerate(sentences):
        if i in breakpoints and current_len >= min_chunk_chars:
            chunks.append(" ".join(current))
            current, current_len = [sentence], len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))

    # Safety net: split any chunk that still exceeds max_chunk_chars
    fallback = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_chars,
        chunk_overlap=0,
        separators=_FALLBACK_SEPARATORS,
        is_separator_regex=True,
    )

    result: list[str] = []
    for chunk in chunks:
        if len(chunk) > max_chunk_chars:
            result.extend(fallback.split_text(chunk))
        else:
            result.append(chunk)

    return [c for c in result if c.strip()]


def character_chunk(
    text: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 150,
) -> list[str]:
    """Split text into fixed-size chunks with overlap using recursive separators."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_FALLBACK_SEPARATORS,
        is_separator_regex=True,
    )
    return splitter.split_text(text)
