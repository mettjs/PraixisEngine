"""Conversation compaction: folds older exchanges into an LLM-written summary.

Sessions are bounded by CONTEXT_WINDOW (a token budget, estimated at ~4 chars
per token) instead of a fixed message count. When a session approaches the
budget, everything except the most recent exchanges is replaced by a single
summary message, so long conversations keep their context instead of silently
losing their oldest turns.

The summarization call here uses the raw client WITHOUT acquiring a GPU slot —
the caller must already hold one (the chat/RAG controllers acquire it before
their stream starts; the standalone compact endpoint acquires its own).
"""
from src.config import CONTEXT_WINDOW as _CONTEXT_WINDOW, MODEL_NAME as _MODEL_NAME
from src.utils.ai_client import get_async_ai_client, record_llm_usage

_client = get_async_ai_client()

# Rough, language-agnostic token estimate used against CONTEXT_WINDOW.
_CHARS_PER_TOKEN = 4
# Compact when the history estimate crosses this fraction of CONTEXT_WINDOW,
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


def estimate_tokens(history: list[dict[str, str]]) -> int:
    return sum(len(m.get("content", "")) for m in history) // _CHARS_PER_TOKEN


def needs_compaction(history: list[dict[str, str]]) -> bool:
    return estimate_tokens(history) >= int(_CONTEXT_WINDOW * _TRIGGER_RATIO)


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
) -> list[dict[str, str]]:
    """Returns a compacted copy of ``history``; the original is not mutated.

    If there is nothing to fold (conversation still fits in the verbatim
    window), the history is returned unchanged. The caller persists the result
    and must hold a GPU slot.
    """
    preamble, older, recent = _split_history(history)
    if not any(m.get("role") != "system" for m in older):
        return history

    transcript = "\n\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in older
    )
    response = await _client.chat.completions.create(
        model=_MODEL_NAME,
        messages=[
            {"role": "system", "content": _SUMMARIZE_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],  # type: ignore[arg-type]
    )
    await record_llm_usage(response, app_name, session_id=session_id)
    summary = response.choices[0].message.content
    if not summary:
        raise RuntimeError("Compaction LLM call returned no content.")

    summary_message = {
        "role": "system",
        "content": f"{_SUMMARY_PREFIX}\nThe earlier part of this conversation was summarized:\n{summary.strip()}",
    }
    return preamble + [summary_message] + recent
