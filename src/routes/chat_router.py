from fastapi import APIRouter, Depends, UploadFile, File, Form
from src.dependencies.security import verify_api_key
from src.models.schemas import ChatRequest
from src.controllers.chat_controller import (
    handle_chat,
    handle_clear_history,
    handle_compact_session,
    handle_fetch_history,
    handle_file_summary,
    handle_list_sessions,
    handle_session_usage,
    handle_undo_last_exchange,
)
from src.utils.system.limiter import rate_limit

router = APIRouter(
    prefix="/general-requests",
    tags=["Core AI Endpoints"],
    dependencies=[Depends(verify_api_key)]
)


@router.post("/chat", dependencies=[Depends(rate_limit("10/minute"))])
async def chat_endpoint(
    chat_request: ChatRequest,
    app_name: str = Depends(verify_api_key)
):
    return await handle_chat(request=chat_request, app_name=app_name)


@router.post("/file_summary", dependencies=[Depends(rate_limit("5/minute"))])
async def file_summary_endpoint(
    file: UploadFile = File(...),
    task: str = Form(default="Summarize the key points of this document."),
    tone: str = Form(default="Professional and objective"),
    stream: bool = Form(default=True, description="Stream tokens as text/event-stream, or return one buffered JSON body."),
    response_format: str = Form(default="text", pattern=r"^(text|json)$", description="LLM content format: 'text' or 'json'."),
    app_name: str = Depends(verify_api_key)
):
    return await handle_file_summary(
        file=file, task=task, tone=tone, app_name=app_name, stream=stream, response_format=response_format,
    )


@router.get("/chat/sessions/active", dependencies=[Depends(rate_limit("60/minute"))])
async def list_active_sessions(app_name: str = Depends(verify_api_key)):
    return await handle_list_sessions(app_name=app_name)


@router.get("/chat/{session_id}", dependencies=[Depends(rate_limit("60/minute"))])
async def fetch_chat_history(session_id: str, app_name: str = Depends(verify_api_key)):
    return await handle_fetch_history(session_id, app_name=app_name)


@router.get("/chat/{session_id}/usage", dependencies=[Depends(rate_limit("60/minute"))])
async def fetch_session_usage(session_id: str, app_name: str = Depends(verify_api_key)):
    return await handle_session_usage(session_id, app_name=app_name)


@router.post("/chat/{session_id}/compact", dependencies=[Depends(rate_limit("10/minute"))])
async def compact_session(session_id: str, app_name: str = Depends(verify_api_key)):
    return await handle_compact_session(session_id, app_name=app_name)


@router.delete("/chat/{session_id}/last", dependencies=[Depends(rate_limit("30/minute"))])
async def undo_last_exchange(session_id: str, app_name: str = Depends(verify_api_key)):
    """Removes the last user message and the assistant reply that followed it,
    so the client can retry or regenerate the exchange."""
    return await handle_undo_last_exchange(session_id, app_name=app_name)


@router.delete("/chat/{session_id}", dependencies=[Depends(rate_limit("30/minute"))])
async def clear_chat_history(session_id: str, app_name: str = Depends(verify_api_key)):
    return await handle_clear_history(session_id, app_name=app_name)
