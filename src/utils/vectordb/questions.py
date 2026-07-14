"""Hypothetical-question generation for the question index.

For each stored chunk, an LLM produces a handful of questions a non-expert would
ask that the chunk answers. Those questions are embedded and handed to the active
backend's ``store_questions`` (pgvector: ``chunk_questions`` table; Chroma: a
parallel questions collection). At query time the user's question is matched
against these (question-to-question, same register) instead of against the formal
source text, closing the genre gap.

Generation and scheduling are fully backend-agnostic; only storage and
question-search live behind the :class:`~src.utils.vectordb.base.VectorStore`
interface.

This module is best-effort by design: it runs in a deferred background pass, so a
single chunk that fails generation is logged and skipped rather than aborting the
whole file (unlike ``llm_runner.map_calls``, which cancels the batch on any error).
"""
import asyncio
import re

from src.config import (
    HQ_ENABLED as _HQ_ENABLED,
    HQ_PER_CHUNK as _PER_CHUNK,
    HQ_GPU_CONCURRENCY as _HQ_SLOTS,
)
from src.services.llm_runner import call_llm
from src.utils.concurrency import hq_gpu_slot
from src.utils.store.client import redis_client
from src.utils.vectordb import get_vector_store
from src.utils.vectordb.base import StaleChunksError
from src.utils.vectordb.embeddings import embed
from src.utils.system.logger import logger

# Bound in-process generation to the size of the reserved GPU pool so we don't
# spawn coroutines that just block waiting on Redis tokens. The Redis HQ queue
# remains the cross-process source of truth; this is the local mirror. Falls back
# to 1 when the dedicated pool is disabled (HQ_GPU_CONCURRENCY=0).
_LOCAL_CONCURRENCY = max(_HQ_SLOTS, 1)

# The prompt is deliberately language- and domain-agnostic: the corpus may be
# legal, medical, administrative, in any language. The one hard requirement is
# register — questions must sound like an ordinary person, not a restatement of
# the source, because that is the entire point of the index.
_SYSTEM_PROMPT = (
    "You help ordinary people find answers inside formal or technical documents. "
    "Given a passage, write the questions a NON-EXPERT would actually ask that "
    "this passage answers. Rules:\n"
    "- Use everyday, plain language — the way a regular person speaks, NOT the "
    "formal wording of the passage.\n"
    "- Write in the SAME LANGUAGE as the passage.\n"
    "- Each question must be answerable from the passage alone.\n"
    "- Output ONE question per line. No numbering, no bullets, no extra text."
)


# Matches a leading list marker ("- ", "• ", "* ", "1. ", "2) ", "(3) ") and
# nothing else — unlike a str.lstrip charset, it can't eat meaningful leading
# digits from a question that genuinely starts with a number or year.
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-•*]+|\(?\d+[.)])\s*")


def _parse_questions(raw: str, limit: int) -> list[str]:
    """Turn the LLM's line-separated output into a clean question list."""
    out: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        q = _LIST_MARKER_RE.sub("", line).strip()
        if len(q) < 8:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= limit:
            break
    return out


async def _questions_for_chunk(content: str, app_name: str, per_chunk: int) -> list[str]:
    """Generate questions for a single chunk. Returns [] on any failure."""
    prompt = (
        f"Write up to {per_chunk} questions a non-expert would ask that the "
        f"following passage answers.\n\nPassage:\n{content}"
    )
    try:
        raw = await call_llm(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            app_name,
            gpu_ctx=hq_gpu_slot,
        )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(f"Question generation failed for a chunk (app={app_name}): {e}")
        return []
    return _parse_questions(raw, per_chunk)


async def generate_and_store_questions(
    app_name: str,
    collection_name: str,
    source: str,
    chunks: list[dict],
    per_chunk: int = _PER_CHUNK,
    purge_existing: bool = False,
) -> int:
    """Generate, embed, and store hypothetical questions for a file's chunks.

    ``chunks`` is the list of just-inserted parent rows, each a dict with keys
    ``id``, ``chunk_index``, ``content``. Returns the number of questions stored.

    ``purge_existing`` (the regenerate path) deletes the file's current
    questions only once the new pass has produced results — a fully-failed pass
    must never leave a file that had a working index with no questions at all.

    In-process generation is bounded by ``_LOCAL_CONCURRENCY`` (the reserved GPU
    pool size) and tolerant of per-chunk failures; a chunk that yields no
    questions is simply skipped.
    """
    if per_chunk <= 0 or not chunks:
        return 0

    sem = asyncio.Semaphore(_LOCAL_CONCURRENCY)

    async def _bounded(chunk: dict) -> list[str]:
        async with sem:
            return await _questions_for_chunk(chunk["content"], app_name, per_chunk)

    questions_per_chunk = await asyncio.gather(
        *(_bounded(c) for c in chunks), return_exceptions=True
    )

    # Flatten into (parent_chunk, question_text) pairs, dropping failures.
    pairs: list[tuple[dict, str]] = []
    for chunk, result in zip(chunks, questions_per_chunk):
        if isinstance(result, BaseException):
            logger.warning(f"Question batch error for chunk {chunk['id']}: {result}")
            continue
        for q in result:
            pairs.append((chunk, q))

    if not pairs:
        logger.info(f"No questions generated for '{source}' (app={app_name}, collection={collection_name}).")
        return 0

    embeddings = await asyncio.to_thread(embed, [q for _, q in pairs])

    entries = [(chunk, q, emb) for (chunk, q), emb in zip(pairs, embeddings)]
    if purge_existing:
        # Replace the old index only now that the new one is ready to store.
        await get_vector_store().delete_questions(app_name, collection_name, source)
    await get_vector_store().store_questions(app_name, collection_name, source, entries)
    logger.info(
        f"Stored {len(entries)} hypothetical questions for '{source}' "
        f"(app={app_name}, collection={collection_name})."
    )
    return len(entries)


# Strong references to in-flight background tasks. asyncio only holds a weak
# reference to a task, so without this the generation coroutine could be garbage
# collected mid-run; the done callback drops the reference once it finishes.
_background_tasks: set[asyncio.Task] = set()

# Generation-in-progress markers live in Redis so status is visible across
# workers. Best-effort: the TTL is a generous ceiling that clears the marker
# even if the owning process dies mid-generation.
_PENDING_TTL = 2 * 3600


def _pending_key(app_name: str, collection_name: str, source: str) -> str:
    return f"hyq:pending:{app_name}:{collection_name}:{source}"


async def is_generation_pending(app_name: str, collection_name: str, source: str) -> bool:
    """True while a background generation pass for this file is running."""
    return bool(await redis_client.exists(_pending_key(app_name, collection_name, source)))


async def _set_pending(app_name: str, collection_name: str, source: str, exclusive: bool) -> bool:
    """Sets the pending marker; with ``exclusive``, only if it is not already set.

    The marker is set BEFORE the background task is created (not inside it), so
    a status check immediately after scheduling sees the pass as pending and an
    exclusive set doubles as an atomic already-running guard.
    """
    key = _pending_key(app_name, collection_name, source)
    if exclusive:
        return bool(await redis_client.set(key, "1", ex=_PENDING_TTL, nx=True))
    await redis_client.set(key, "1", ex=_PENDING_TTL)
    return True


async def _run_generation(
    app_name: str, collection_name: str, source: str, chunks: list[dict], purge_existing: bool = False
) -> None:
    """Background wrapper: best-effort, never raises into the event loop.

    The pending marker is already set by the scheduler; this only clears it.
    """
    pending_key = _pending_key(app_name, collection_name, source)
    try:
        await generate_and_store_questions(
            app_name, collection_name, source, chunks, purge_existing=purge_existing
        )
    except StaleChunksError:
        # The file was deleted or re-uploaded while we were generating, so the
        # parent chunks these questions point to no longer exist. The newer
        # upload will regenerate; nothing to recover here.
        logger.info(
            f"Skipped question generation for '{source}' (app={app_name}): "
            f"parent chunks superseded by a newer upload."
        )
    except Exception as e:
        logger.error(f"Background question generation failed for '{source}' (app={app_name}): {e}")
    finally:
        try:
            await redis_client.delete(pending_key)
        except Exception:
            pass  # the TTL clears it eventually


async def schedule_question_generation(
    app_name: str,
    collection_name: str,
    source: str,
    chunks: list[dict],
    *,
    exclusive: bool = False,
    purge_existing: bool = False,
) -> bool:
    """Fire-and-forget hypothetical-question generation for a just-stored file.

    Sets the pending marker, then returns; generation runs on the event loop
    without blocking the upload response. With ``exclusive``, nothing is
    scheduled when a pass for this file is already pending (the regenerate
    endpoint's already-running guard — the marker set is atomic, so concurrent
    callers cannot both schedule). ``purge_existing`` is forwarded to the pass
    (see :func:`generate_and_store_questions`).

    Returns True when a pass was scheduled; False when generation is disabled,
    there is nothing to do, or an exclusive schedule found a pass already
    pending.
    """
    if not _HQ_ENABLED or not chunks or not get_vector_store().supports_questions:
        return False
    if not await _set_pending(app_name, collection_name, source, exclusive):
        return False
    task = asyncio.create_task(
        _run_generation(app_name, collection_name, source, chunks, purge_existing=purge_existing)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return True
