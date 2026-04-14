from src.models.schemas import ChatRequest
from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from src.utils.file_parser import extract_text_from_file
from src.services.chat_service import generate_chat_stream, generate_file_summary
from src.utils.memory import delete_session, get_all_active_sessions, get_session_history
from src.utils.logger import logger
from src.utils.concurrency import GPUBusyError, acquire_gpu_slot, release_gpu_slot

_MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

def handle_chat(request: ChatRequest, app_name: str) -> StreamingResponse:
    # Acquire before StreamingResponse so we can still return a 503 if the GPU is full.
    # Once headers are sent it's too late to change the status code.
    try:
        acquire_gpu_slot()
    except GPUBusyError as e:
        raise HTTPException(status_code=503, detail=str(e))

    def _stream():
        try:
            yield from generate_chat_stream(
                app_name=app_name,
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                session_id=request.session_id,
            )
        finally:
            release_gpu_slot()

    try:
        logger.info(f"Received chat request for app: {app_name}, session: {request.session_id}")
        return StreamingResponse(_stream(), media_type="text/event-stream")
    except Exception as e:
        release_gpu_slot()
        logger.error(f"Error in handle_chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def handle_file_summary(file: UploadFile, task: str, tone: str, app_name: str) -> dict:
    if not file.filename:
        logger.warning("Received file summary request without a file.")
        raise HTTPException(status_code=400, detail="No file uploaded.")
    try:
        content = await file.read(_MAX_FILE_SIZE + 1)
        if len(content) > _MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum allowed size is 20 MB.")

        document_text = extract_text_from_file(file.filename, content)

        if not document_text.strip():
            raise ValueError("Could not extract any text from the provided file.")
        
        result = generate_file_summary(document_text=document_text, task=task, tone=tone) 
        
        logger.info(f"Generated file summary for app: {app_name}")
        
        return {
            "status": "success", 
            "filename": file.filename,
            "task_executed": task,
            "tone_applied": tone,
            "result": result
        }
    except HTTPException:
        raise
    except GPUBusyError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as ve:
        logger.warning(f"Value error in handle_file_summary: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in handle_file_summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
def handle_fetch_history(session_id: str, app_name: str) -> dict:
    history = get_session_history(session_id=session_id, app_name=app_name)
    logger.info(f"Fetched chat history for app: {app_name}, session: {session_id}, messages: {len(history)}")
    if not history:
        logger.warning(f"Session not found or expired for app: {app_name}, session: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    return {"session_id": session_id, "history": history}

def handle_clear_history(session_id: str, app_name: str) -> dict:
    success = delete_session(session_id=session_id, app_name=app_name)
    logger.info(f"Deleted chat history for app: {app_name}, session: {session_id}")
    if not success:
        logger.warning(f"Session not found for deletion for app: {app_name}, session: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"status": "success", "detail": "Session deleted."}

def handle_list_sessions(app_name: str) -> dict:
    logger.info(f"Listing active sessions for app: {app_name}")
    return {"active_sessions": get_all_active_sessions(app_name=app_name)}