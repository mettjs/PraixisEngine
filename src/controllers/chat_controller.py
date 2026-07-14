from src.models.schemas import ChatRequest
from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from src.utils.file_parser import extract_text_from_file, MAX_FILE_SIZE
from src.services.chat_service import generate_chat_stream, generate_file_summary
from src.services.compaction import compact_history, compactable, estimate_tokens
from src.utils.store.sessions import delete_session, get_all_active_sessions, get_session_history, persist_history
from src.utils.store.usage import get_session_usage
from src.utils.system.logger import logger
from src.utils.system.streaming import SlotReleasingStreamingResponse, drain_to_json, guard_stream
from src.utils.concurrency import GPUBusyError, acquire_gpu_slot


async def handle_chat(request: ChatRequest, app_name: str) -> StreamingResponse | dict:
    try:
        slot = await acquire_gpu_slot()
    except GPUBusyError as e:
        raise HTTPException(status_code=503, detail=str(e))

    reply = generate_chat_stream(
        app_name=app_name,
        prompt=request.prompt,
        system_prompt=request.system_prompt,
        session_id=request.session_id,
        response_format=request.response_format,
    )

    # The streaming wrapper owns the slot and releases it whether the stream
    # completes, errors, or the client disconnects before it even starts. The
    # buffered path drains the same generator and must release the slot itself.
    # If anything fails before either takes over, release here.
    try:
        if request.stream:
            logger.info(f"Received chat request for app: {app_name}, session: {request.session_id}")
            return SlotReleasingStreamingResponse(guard_stream(reply), slot=slot, media_type="text/event-stream")
        logger.info(f"Received buffered chat request for app: {app_name}, session: {request.session_id}")
        try:
            return await drain_to_json(reply)
        finally:
            await slot.release()
    except Exception:
        await slot.release()
        raise


async def handle_file_summary(
    file: UploadFile,
    task: str,
    tone: str,
    app_name: str,
    stream: bool = True,
    response_format: str = "text",
) -> StreamingResponse | dict:
    if not file.filename:
        logger.warning("Received file summary request without a file.")
        raise HTTPException(status_code=400, detail="No file uploaded.")

    content = await file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum allowed size is 20 MB.")

    try:
        document_text = extract_text_from_file(file.filename, content, content_type=file.content_type)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    if not document_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from the provided file.")

    filename = file.filename

    # generate_file_summary manages its own GPU slots per LLM call via the shared
    # runner, so we must NOT pre-acquire a slot here (doing so would deadlock
    # under concurrency).
    async def _summary_with_header():
        yield f"[FILE:{filename}]\n"
        async for token in generate_file_summary(
            document_text=document_text,
            task=task,
            tone=tone,
            app_name=app_name,
            response_format=response_format,
        ):
            yield token

    # Buffered mode hasn't sent headers yet, so an exhausted-GPU error can be a
    # real 503. The streaming response has already started, so the same failure
    # can only be surfaced as an in-stream error marker.
    if not stream:
        logger.info(f"Buffering file summary for app: {app_name}, file: {filename}")
        try:
            return await drain_to_json(_summary_with_header())
        except GPUBusyError as e:
            raise HTTPException(status_code=503, detail=str(e))

    logger.info(f"Streaming file summary for app: {app_name}, file: {filename}")
    return StreamingResponse(guard_stream(_summary_with_header()), media_type="text/event-stream")


async def handle_fetch_history(session_id: str, app_name: str) -> dict:
    history = await get_session_history(session_id=session_id, app_name=app_name)
    logger.info(f"Fetched chat history for app: {app_name}, session: {session_id}, messages: {len(history)}")
    if not history:
        logger.warning(f"Session not found or expired for app: {app_name}, session: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    return {"session_id": session_id, "history": history}


async def handle_clear_history(session_id: str, app_name: str) -> dict:
    success = await delete_session(session_id=session_id, app_name=app_name)
    logger.info(f"Deleted chat history for app: {app_name}, session: {session_id}")
    if not success:
        logger.warning(f"Session not found for deletion for app: {app_name}, session: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"status": "success", "detail": "Session deleted."}


async def handle_undo_last_exchange(session_id: str, app_name: str) -> dict:
    """Removes the last user message and everything after it (the assistant
    reply — or nothing, if generation failed), so the client can retry or
    regenerate. System messages, including compaction summaries, stay."""
    history = await get_session_history(session_id=session_id, app_name=app_name)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    user_indices = [i for i, m in enumerate(history) if m.get("role") == "user"]
    if not user_indices:
        raise HTTPException(status_code=400, detail="Nothing to undo — the session has no user messages left.")
    cut = user_indices[-1]
    removed = history[cut:]
    remaining = history[:cut]
    await persist_history(app_name=app_name, session_id=session_id, history=remaining)
    logger.info(
        f"Undid last exchange for session {session_id} (app: {app_name}): removed {len(removed)} message(s)."
    )
    return {
        "status": "success",
        "session_id": session_id,
        "removed_messages": len(removed),
        "undone_prompt": removed[0].get("content", ""),
        "messages_remaining": len(remaining),
    }


async def handle_list_sessions(app_name: str) -> dict:
    logger.info(f"Listing active sessions for app: {app_name}")
    return {"active_sessions": await get_all_active_sessions(app_name=app_name)}


async def handle_session_usage(session_id: str, app_name: str) -> dict:
    history = await get_session_history(session_id=session_id, app_name=app_name)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    usage = await get_session_usage(app_name=app_name, session_id=session_id)
    usage["estimated_context_tokens"] = estimate_tokens(history)
    return usage


async def handle_compact_session(session_id: str, app_name: str) -> dict:
    history = await get_session_history(session_id=session_id, app_name=app_name)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    if not compactable(history):
        raise HTTPException(
            status_code=400,
            detail="Not enough history to compact — the conversation already fits in the verbatim window.",
        )

    try:
        slot = await acquire_gpu_slot()
    except GPUBusyError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # compact_history uses the raw client, so this endpoint owns the slot for
    # the duration of the summarization call.
    try:
        compacted = await compact_history(history, app_name=app_name, session_id=session_id)
    except Exception as e:
        logger.error(f"Compaction failed for session {session_id} (app: {app_name}): {e}")
        raise HTTPException(status_code=502, detail="Compaction failed: could not summarize the conversation.")
    finally:
        await slot.release()

    await persist_history(app_name=app_name, session_id=session_id, history=compacted)
    logger.info(
        f"Compacted session {session_id} (app: {app_name}): "
        f"{len(history)} -> {len(compacted)} messages"
    )
    return {
        "status": "success",
        "session_id": session_id,
        "messages_before": len(history),
        "messages_after": len(compacted),
        "estimated_tokens_before": estimate_tokens(history),
        "estimated_tokens_after": estimate_tokens(compacted),
    }
