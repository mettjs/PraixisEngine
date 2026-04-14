from fastapi import APIRouter
from src.routes.chat_router import router as general_router
from src.routes.rag_router import router as vector_router
from src.routes.admin_router import router as admin_router

api_router = APIRouter()

api_router.include_router(general_router)
api_router.include_router(vector_router)
api_router.include_router(admin_router)
