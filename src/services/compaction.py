"""Conversation compaction: folds older exchanges into an LLM-written summary.

Sessions are bounded by the chat model's context window (a token budget,
estimated at ~4 chars per token) instead of a fixed message count. When a session approaches the
budget, everything except the most recent exchanges is replaced by a single
summary message, so long conversations keep their context instead of silently
losing their oldest turns.

Two models are in play, and they are deliberately different: the budget is
measured against the **chat** model's context window (that model is the one
that has to swallow the history), while the summarization call itself runs on
the **utility** role, so a chat routed to an expensive model does not bill its
housekeeping there too.

That split is also why the caller must say which pool it already holds. The
summarization call needs capacity on the *utility* model's backend, and the
slot the caller is holding only covers the chat model's. When the two pools
coincide — every single-pool deployment — reusing the held slot is both correct
and necessary: acquiring a second token from the same bucket while holding one
is a deadlock under load. When they differ, the held slot buys nothing on the
utility backend and this module takes its own.
"""
from src.models.registry import ModelSpec, resolve_role
from src.utils.ai_client import get_async_ai_client, record_llm_usage
from src.utils.concurrency import gpu_slot

# Rough, language-agnostic token estimate used against the model's window.
_CHARS_PER_TOKEN = 4
# Compact when the history estimate crosses this fraction of that window,
# leaving the remainder as headroom for the completion (and RAG context).
_TRIGGER_RATIO = 0.8
# Most recent user+assistant pairs kept verbatim through a compaction.
_KEEP_RECENT_PAIRS = 3

_SUMMARY_PREFIX = "[CONVERSATION SUMMARY]"

_SUMMARIZE_SYSTEM_PROMPT = (
    "You are a conversation summarizer. Condense the following conversation "
    "transcript into a dense summary that preserves everything needed to "
    "continue the conversation seamlessly: stated facts, names, numbers, "
    "decisions made, user preferences and constraints, unresolved questions, "
    "and the overall goal. Write it in the transcript's language. "
    "Return ONLY the summary, no preamble."
)


async def _summarize(spec: ModelSpec, transcript: str):
    return await get_async_ai_client(spec).chat.completions.create(
        model=spec.model,
        messages=[
            {"role": "system", "content": _SUMMARIZE_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],  # type: ignore[arg-type]
        **spec.params,
    )


def estimate_tokens(history: list[dict[str, str]]) -> int:
    return sum(len(m.get("content", "")) for m in history) // _CHARS_PER_TOKEN


def needs_compaction(history: list[dict[str, str]], spec: ModelSpec) -> bool:
    """``spec`` must be the model that will *consume* this history, not the
    utility model that writes the summary."""
    return estimate_tokens(history) >= int(spec.context_window * _TRIGGER_RATIO)


def _split_history(
    history: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Splits history into (preamble, older, recent).

    preamble: leading system message(s) minus any prior summary; older: prior
    summary (folded in so its context carries forward) plus the messages to
    summarize; recent: the last _KEEP_RECENT_PAIRS exchanges kept verbatim,
    aligned to start on a user message.
    """
    rest = list(history)
    preamble: list[dict[str, str]] = []
    while rest and rest[0].get("role") == "system":
        preamble.append(rest.pop(0))
    prior_summary = [m for m in preamble if m["content"].startswith(_SUMMARY_PREFIX)]
    preamble = [m for m in preamble if not m["content"].startswith(_SUMMARY_PREFIX)]

    cut = max(len(rest) - _KEEP_RECENT_PAIRS * 2, 0)
    while cut > 0 and rest[cut].get("role") != "user":
        cut -= 1
    return preamble, prior_summary + rest[:cut], rest[cut:]


def compactable(history: list[dict[str, str]]) -> bool:
    """True when the history has older exchanges that a compaction would fold."""
    _, older, _ = _split_history(history)
    return any(m.get("role") != "system" for m in older)


async def compact_history(
    history: list[dict[str, str]],
    app_name: str,
    session_id: str,
    held_pool: str,
) -> list[dict[str, str]]:
    """Returns a compacted copy of ``history``; the original is not mutated.

    If there is nothing to fold (conversation still fits in the verbatim
    window), the history is returned unchanged. The summary is written by the
    ``utility`` role's model. The caller persists the result.

    ``held_pool`` is the pool the caller already holds a slot in. A slot is
    taken here only when the utility model draws from a *different* pool — see
    the module docstring for why reusing the caller's slot is required when
    they match.
    """
    preamble, older, recent = _split_history(history)
    if not any(m.get("role") != "system" for m in older):
        return history

    transcript = "\n\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in older
    )
    spec = resolve_role("utility")
    if spec.pool == held_pool:
        # Same bucket: the caller's slot already covers this call.
        response = await _summarize(spec, transcript)
    else:
        async with gpu_slot(spec.pool):
            response = await _summarize(spec, transcript)
    await record_llm_usage(response, app_name, session_id=session_id, model_id=spec.id)
    summary = response.choices[0].message.content
    if not summary:
        raise RuntimeError("Compaction LLM call returned no content.")

    summary_message = {
        "role": "system",
        "content": f"{_SUMMARY_PREFIX}\nThe earlier part of this conversation was summarized:\n{summary.strip()}",
    }
    return preamble + [summary_message] + recent
