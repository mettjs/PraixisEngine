from src.models.schemas import ChatRequest
from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from src.utils.file_parser import extract_text_from_file
from src.services.chat_service import generate_chat_stream, generate_file_summary
from src.utils.memory import delete_session, get_all_active_sessions, get_session_history
from src.utils.logger import logger
from src.utils.concurrency import GPUBusyError, acquire_gpu_slot, release_gpu_slot

_MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


async def handle_chat(request: ChatRequest, app_name: str) -> StreamingResponse:
    try:
        await acquire_gpu_slot()
    except GPUBusyError as e:
        raise HTTPException(status_code=503, detail=str(e))

    async def _stream():
        try:
            async for token in generate_chat_stream(
                app_name=app_name,
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                session_id=request.session_id,
                response_format=request.response_format,
            ):
                yield token
        finally:
            await release_gpu_slot()

    try:
        logger.info(f"Received chat request for app: {app_name}, session: {request.session_id}")
        return StreamingResponse(_stream(), media_type="text/event-stream")
    except Exception as e:
        await release_gpu_slot()
        logger.error(f"Error in handle_chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def handle_file_summary(file: UploadFile, task: str, tone: str, app_name: str) -> StreamingResponse:
    if not file.filename:
        logger.warning("Received file summary request without a file.")
        raise HTTPException(status_code=400, detail="No file uploaded.")

    content = await file.read(_MAX_FILE_SIZE + 1)
    if len(content) > _MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum allowed size is 20 MB.")

    try:
        document_text = extract_text_from_file(file.filename, content)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    if not document_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from the provided file.")

    try:
        await acquire_gpu_slot()
    except GPUBusyError as e:
        raise HTTPException(status_code=503, detail=str(e))

    filename = file.filename

    async def _stream():
        try:
            yield f"[FILE:{filename}]\n"
            async for token in generate_file_summary(
                document_text=document_text,
                task=task,
                tone=tone,
                app_name=app_name,
            ):
                yield token
        finally:
            await release_gpu_slot()

    logger.info(f"Streaming file summary for app: {app_name}, file: {filename}")
    return StreamingResponse(_stream(), media_type="text/event-stream")


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
