from typing import List
from fastapi import APIRouter, Depends, File, Path, Request, UploadFile, Form
from src.dependencies.security import verify_api_key
from src.models.schemas import EmbedRequest, QuestionRequest
from src.controllers.rag_controller import (
    handle_rag_upload,
    handle_compare_documents,
    handle_delete_file,
    handle_embed,
    handle_list_files,
    handle_rag_question,
    handle_list_collections,
    handle_delete_collection,
    handle_summarize_document,
)
from src.utils.limiter import limiter

router = APIRouter(
    prefix="/rag-db",
    tags=["Vector optimized Endpoints"],
    dependencies=[Depends(verify_api_key)]
)


@router.post("/upload")
@limiter.limit("15/minute")
async def rag_upload_endpoint(
    request: Request,
    files: List[UploadFile] = File(..., description="One or more .pdf, .docx, or .txt files — max 20 MB each."),
    collection_name: str = Form(default="main", pattern=r"^[a-zA-Z0-9_-]{3,63}$",
                                description="Target collection name. Defaults to 'main'."),
    chunk_size: int = Form(default=1000, ge=100, le=4000, description="Characters per chunk (100–4000)."),
    chunk_overlap: int = Form(default=150, ge=0, le=500, description="Overlap characters between chunks (0–500)."),
    app_name: str = Depends(verify_api_key)
):
    return await handle_rag_upload(
        collection_name=collection_name,
        files=files,
        app_name=app_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


@router.post("/ask")
@limiter.limit("30/minute")
async def rag_ask_endpoint(
    request: Request,
    question_request: QuestionRequest,
    app_name: str = Depends(verify_api_key)
):
    return await handle_rag_question(question_request, app_name=app_name)


@router.post("/embed")
@limiter.limit("60/minute")
async def embed_endpoint(
    request: Request,
    embed_request: EmbedRequest,
):
    return await handle_embed(embed_request)


@router.get("/list")
async def rag_list_endpoint(app_name: str = Depends(verify_api_key)):
    return await handle_list_collections(app_name=app_name)


@router.get("/{collection_name}/files")
async def rag_list_files_endpoint(
    collection_name: str = Path(..., pattern=r"^[a-zA-Z0-9_-]{3,63}$"),
    app_name: str = Depends(verify_api_key)
):
    return await handle_list_files(collection_name=collection_name, app_name=app_name)


@router.delete("/delete/{collection_name}")
@limiter.limit("20/minute")
async def rag_delete_endpoint(
    request: Request,
    collection_name: str,
    app_name: str = Depends(verify_api_key)
):
    return await handle_delete_collection(collection_name=collection_name, app_name=app_name)


@router.delete("/{collection_name}/files/{filename}")
@limiter.limit("20/minute")
async def rag_delete_file_endpoint(
    request: Request,
    filename: str = Path(..., description="Exact filename to delete (e.g. policy_2024.pdf)."),
    collection_name: str = Path(..., pattern=r"^[a-zA-Z0-9_-]{3,63}$"),
    app_name: str = Depends(verify_api_key)
):
    return await handle_delete_file(collection_name=collection_name, filename=filename, app_name=app_name)


@router.post("/knowledge_base/compare")
@limiter.limit("5/minute")
async def rag_compare_documents(
    request: Request,
    collection_name: str,
    file_1: str,
    file_2: str,
    app_name: str = Depends(verify_api_key)
):
    return await handle_compare_documents(collection_name=collection_name, file_1=file_1, file_2=file_2, app_name=app_name)


@router.get("/knowledge_base/{collection_name}/files/{filename}/summary")
@limiter.limit("10/minute")
async def rag_summarize_document(
    request: Request,
    collection_name: str,
    filename: str,
    app_name: str = Depends(verify_api_key)
):
    return await handle_summarize_document(collection_name=collection_name, filename=filename, app_name=app_name)
