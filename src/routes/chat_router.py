from fastapi import APIRouter, Depends, Request, UploadFile, File, Form
from src.dependencies.security import verify_api_key
from src.models.schemas import ChatRequest
from src.controllers.chat_controller import handle_chat, handle_clear_history, handle_fetch_history, handle_file_summary, handle_list_sessions
from src.utils.limiter import limiter

router = APIRouter(
    prefix="/general-requests",
    tags=["Core AI Endpoints"],
    dependencies=[Depends(verify_api_key)]
)


@router.post("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(
    request: Request,
    chat_request: ChatRequest,
    app_name: str = Depends(verify_api_key)
):
    return await handle_chat(request=chat_request, app_name=app_name)


@router.post("/file_summary")
@limiter.limit("5/minute")
async def file_summary_endpoint(
    request: Request,
    file: UploadFile = File(...),
    task: str = Form(default="Summarize the key points of this document."),
    tone: str = Form(default="Professional and objective"),
    app_name: str = Depends(verify_api_key)
):
    return await handle_file_summary(file=file, task=task, tone=tone, app_name=app_name)


@router.get("/chat/sessions/active")
async def list_active_sessions(app_name: str = Depends(verify_api_key)):
    return await handle_list_sessions(app_name=app_name)


@router.get("/chat/{session_id}")
async def fetch_chat_history(session_id: str, app_name: str = Depends(verify_api_key)):
    return await handle_fetch_history(session_id, app_name=app_name)


@router.delete("/chat/{session_id}")
async def clear_chat_history(session_id: str, app_name: str = Depends(verify_api_key)):
    return await handle_clear_history(session_id, app_name=app_name)
