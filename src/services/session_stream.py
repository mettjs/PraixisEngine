"""Shared session-aware streaming helpers for chat and RAG answers.

Both streaming endpoints follow the same shape: open (or resume) a session,
persist the user's message, stream the completion token-by-token, and persist
the assistant's (possibly partial) reply when the stream ends. The GPU slot is
owned by ``SlotReleasingStreamingResponse`` in the controller, not here.
"""
from collections.abc import AsyncGenerator

from src.config import MODEL_NAME as _MODEL_NAME
from src.services.compaction import compact_history, needs_compaction
from src.utils.ai_client import get_async_ai_client, record_llm_usage
from src.utils.store.sessions import get_or_create_session, persist_history
from src.utils.system.logger import logger

_client = get_async_ai_client()


async def open_user_turn(
    app_name: str,
    user_message: str,
    session_id: str | None = None,
    system_prompt: str | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Resolves the session, appends the user message, and persists it.

    Persisting before the LLM call means the user's message survives even if
    generation fails afterwards. When the history approaches the CONTEXT_WINDOW
    budget it is auto-compacted first — safe here because both callers (chat
    and RAG) already hold the GPU slot the compaction call needs. A compaction
    failure falls back to the uncompacted history rather than failing the turn.
    """
    active_session_id, history = await get_or_create_session(
        session_id=session_id,
        system_prompt=system_prompt,
        app_name=app_name,
    )
    history.append({"role": "user", "content": user_message})
    if needs_compaction(history):
        try:
            history = await compact_history(history, app_name=app_name, session_id=active_session_id)
            logger.info(f"Auto-compacted session {active_session_id} (app: {app_name}).")
        except Exception as e:
            logger.warning(
                f"Auto-compaction failed for session {active_session_id} (app: {app_name}): {e}"
            )
    await persist_history(app_name=app_name, session_id=active_session_id, history=history)
    return active_session_id, history


async def stream_assistant_turn(
    messages: list[dict[str, str]],
    app_name: str,
    session_id: str,
    history: list[dict[str, str]],
    extra: dict | None = None,
) -> AsyncGenerator[str, None]:
    """Streams a completion, then appends the accumulated reply to ``history``
    and persists it — including partial replies when the client disconnects.

    ``messages`` is what the LLM sees; ``history`` is what the session stores.
    RAG passes an augmented copy as ``messages`` so retrieved context is never
    persisted into the conversation. A persistence failure in the ``finally``
    is logged rather than raised so it cannot mask the stream's own outcome.
    """
    full_response = ""
    try:
        response = await _client.chat.completions.create(  # type: ignore[call-overload]
            model=_MODEL_NAME,
            messages=messages,  # type: ignore[arg-type]
            stream=True,
            stream_options={"include_usage": True},
            **(extra or {}),
        )
        usage_recorded = False
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                token = chunk.choices[0].delta.content
                full_response += token
                yield token
            if not usage_recorded and getattr(chunk, "usage", None):
                await record_llm_usage(chunk, app_name, session_id=session_id)
                usage_recorded = True
    finally:
        if full_response:
            history.append({"role": "assistant", "content": full_response})
            try:
                await persist_history(app_name=app_name, session_id=session_id, history=history)
            except Exception as e:
                logger.warning(
                    f"Failed to persist assistant reply for session {session_id} (app: {app_name}): {e}"
                )
