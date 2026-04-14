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
@limiter.limit("10/minute") # Protects the GPU: Max 10 messages per minute per user
def chat_endpoint(
    request: Request,
    chat_request: ChatRequest,
    app_name: str = Depends(verify_api_key)
    ):
    return handle_chat(request=chat_request, app_name=app_name)

@router.post("/file_summary")
@limiter.limit("5/minute") # Protects the GPU: Max 5 file summaries per minute per user
async def file_summary_endpoint(
    request: Request,
    file: UploadFile = File(...),
    task: str = Form(default="Summarize the key points of this document.", description="The task to be performed on the uploaded file. If left blank, it defaults to summarization."),
    tone: str = Form(default="Professional and objective", description="The tone to be used by the AI. If left blank, it defaults to 'Professional and objective'."),
    app_name: str = Depends(verify_api_key)
):
    return await handle_file_summary(file=file, task=task, tone=tone, app_name=app_name)

@router.get("/chat/{session_id}")
def fetch_chat_history(session_id: str, app_name: str = Depends(verify_api_key)):
    return handle_fetch_history(session_id, app_name=app_name)

@router.delete("/chat/{session_id}")
def clear_chat_history(session_id: str, app_name: str = Depends(verify_api_key)):
    return handle_clear_history(session_id, app_name=app_name)

@router.get("/chat/sessions/active")
def list_active_sessions(app_name: str = Depends(verify_api_key)):
    return handle_list_sessions(app_name=app_name)