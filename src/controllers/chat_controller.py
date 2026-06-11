from src.models.schemas import ChatRequest
from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from src.utils.file_parser import extract_text_from_file, MAX_FILE_SIZE
from src.services.chat_service import generate_chat_stream, generate_file_summary
from src.utils.store.sessions import delete_session, get_all_active_sessions, get_session_history
from src.utils.system.logger import logger
from src.utils.system.streaming import SlotReleasingStreamingResponse
from src.utils.concurrency import GPUBusyError, acquire_gpu_slot


async def handle_chat(request: ChatRequest, app_name: str) -> StreamingResponse:
    try:
        slot = await acquire_gpu_slot()
    except GPUBusyError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # The response wrapper owns the slot and releases it whether the stream
    # completes, errors, or the client disconnects before it even starts. If
    # anything fails before the wrapper takes over, release here.
    try:
        logger.info(f"Received chat request for app: {app_name}, session: {request.session_id}")
        return SlotReleasingStreamingResponse(
            generate_chat_stream(
                app_name=app_name,
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                session_id=request.session_id,
                response_format=request.response_format,
            ),
            slot=slot,
            media_type="text/event-stream",
        )
    except Exception:
        await slot.release()
        raise


async def handle_file_summary(file: UploadFile, task: str, tone: str, app_name: str) -> StreamingResponse:
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
    # under concurrency). If every slot stays busy past the timeout, surface it
    # as an in-stream error since the response has already started.
    async def _with_file_header():
        yield f"[FILE:{filename}]\n"
        try:
            async for token in generate_file_summary(
                document_text=document_text,
                task=task,
                tone=tone,
                app_name=app_name,
            ):
                yield token
        except GPUBusyError as e:
            yield f"[ERROR:{e}]\n"

    logger.info(f"Streaming file summary for app: {app_name}, file: {filename}")
    return StreamingResponse(_with_file_header(), media_type="text/event-stream")


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


async def handle_list_sessions(app_name: str) -> dict:
    logger.info(f"Listing active sessions for app: {app_name}")
    return {"active_sessions": await get_all_active_sessions(app_name=app_name)}
