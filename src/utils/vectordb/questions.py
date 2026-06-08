"""Hypothetical-question generation for the question index.

For each stored chunk, an LLM produces a handful of questions a non-expert would
ask that the chunk answers. Those questions are embedded and written to
``chunk_questions``. At query time the user's question is matched against these
(question-to-question, same register) instead of against the formal source text,
closing the genre gap. See ``constants.CREATE_QUESTIONS_SCHEMA``.

This module is best-effort by design: it runs in a deferred background pass, so a
single chunk that fails generation is logged and skipped rather than aborting the
whole file (unlike ``llm_runner.map_calls``, which cancels the batch on any error).
"""
import asyncio
import uuid

import asyncpg

from src.config import (
    HQ_ENABLED as _HQ_ENABLED,
    HQ_PER_CHUNK as _PER_CHUNK,
    HQ_GPU_CONCURRENCY as _HQ_SLOTS,
)
from src.services.llm_runner import call_llm
from src.utils.concurrency import hq_gpu_slot
from src.utils.vectordb.embeddings import embed
from src.utils.vectordb.pool import get_pool
from src.utils.vectordb.constants import INSERT_QUESTION
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


def _parse_questions(raw: str, limit: int) -> list[str]:
    """Turn the LLM's line-separated output into a clean question list."""
    out: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        q = line.strip().lstrip("-•*0123456789.) \t").strip()
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
) -> int:
    """Generate, embed, and store hypothetical questions for a file's chunks.

    ``chunks`` is the list of just-inserted parent rows, each a dict with keys
    ``id``, ``chunk_index``, ``content``. Returns the number of questions stored.

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

    rows = [
        (
            uuid.uuid4().hex,
            app_name,
            collection_name,
            source,
            chunk["id"],
            chunk["chunk_index"],
            q,
            emb,
        )
        for (chunk, q), emb in zip(pairs, embeddings)
    ]

    await get_pool().executemany(INSERT_QUESTION, rows)
    logger.info(
        f"Stored {len(rows)} hypothetical questions for '{source}' "
        f"(app={app_name}, collection={collection_name})."
    )
    return len(rows)


# Strong references to in-flight background tasks. asyncio only holds a weak
# reference to a task, so without this the generation coroutine could be garbage
# collected mid-run; the done callback drops the reference once it finishes.
_background_tasks: set[asyncio.Task] = set()


async def _run_generation(app_name: str, collection_name: str, source: str, chunks: list[dict]) -> None:
    """Background wrapper: best-effort, never raises into the event loop."""
    try:
        await generate_and_store_questions(app_name, collection_name, source, chunks)
    except asyncpg.ForeignKeyViolationError:
        # The file was deleted or re-uploaded while we were generating, so the
        # parent chunks these questions point to no longer exist. The newer
        # upload will regenerate; nothing to recover here.
        logger.info(
            f"Skipped question generation for '{source}' (app={app_name}): "
            f"parent chunks superseded by a newer upload."
        )
    except Exception as e:
        logger.error(f"Background question generation failed for '{source}' (app={app_name}): {e}")


def schedule_question_generation(
    app_name: str,
    collection_name: str,
    source: str,
    chunks: list[dict],
) -> None:
    """Fire-and-forget hypothetical-question generation for a just-stored file.

    Returns immediately; generation runs on the event loop without blocking the
    upload response. A no-op when HQ_ENABLED is false or there is nothing to do.
    """
    if not _HQ_ENABLED or not chunks:
        return
    task = asyncio.create_task(_run_generation(app_name, collection_name, source, chunks))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
