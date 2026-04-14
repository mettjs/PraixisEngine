from fastapi import APIRouter, Depends, File, Path, Request, UploadFile, Form
from src.dependencies.security import verify_api_key
from src.models.schemas import QuestionRequest
from src.controllers.rag_controller import (
    handle_compare_documents,
    handle_delete_file,
    handle_list_files,
    handle_rag_upload,
    handle_rag_question, 
    handle_list_collections,
    handle_delete_collection,
    handle_summarize_document)
from src.utils.limiter import limiter

router = APIRouter(
    prefix="/rag-db", 
    tags=["Vector optimized Endpoints"],
    dependencies=[Depends(verify_api_key)]
    )

@router.post("/upload")
@limiter.limit("15/minute") # Protects the GPU: Max 15 uploads per minute per user
async def rag_upload_endpoint(
    request: Request,
    file: UploadFile = File(...),
    collection_name: str = Form(default="main",
                                pattern=r"^[a-zA-Z0-9_-]{3,63}$",
                                description="""The name of the collection to store the document in. 
                                          If left blank, it defaults to 'main'. 
                                          This should be a string without any special characters.
                                          If a collection with the same name already exists, file will be uploaded to that collection."""),
    app_name: str = Depends(verify_api_key)
    ):
    return await handle_rag_upload(collection_name=collection_name, file=file, app_name=app_name)

@router.post("/ask")
@limiter.limit("30/minute") # Protects the GPU: Max 30 questions per minute per user
def rag_ask_endpoint(    
    request: Request,
    question_request: QuestionRequest,
    app_name: str = Depends(verify_api_key)
    ):
    """Ask a specific question about a previously uploaded collection."""
    return handle_rag_question(question_request, app_name=app_name)

@router.get("/list")
def rag_list_endpoint(app_name: str = Depends(verify_api_key)):
    """See exactly what collections are currently on the server."""
    return handle_list_collections(app_name=app_name)

@router.get("/{collection_name}/files")
def list_files_endpoint(
    collection_name: str = Path(
        ..., 
        pattern=r"^[a-zA-Z0-9_-]{3,63}$",
        description="The exact name of the collection you want to inspect."
    ),
    app_name: str = Depends(verify_api_key)
):
    """See all the unique files currently stored inside a specific collection."""
    return handle_list_files(collection_name=collection_name, app_name=app_name)

@router.delete("/delete/{collection_name}")
@limiter.limit("20/minute") # Protects the GPU: Max 20 delete requests per minute per user
def rag_delete_endpoint(
    request: Request,
    collection_name: str,
    app_name: str = Depends(verify_api_key)
    ):
    """Permanently purge an outdated collections from the AI's memory."""
    return handle_delete_collection(collection_name=collection_name, app_name=app_name)

@router.delete("/{collection_name}/files/{filename}")
@limiter.limit("20/minute") # Protects the GPU: Max 20 delete requests per minute per user
def delete_file_endpoint(
    request: Request,
    filename: str = Path(
        ..., 
        description="The exact name of the file you want to delete (e.g., policy_2024.pdf)."
    ),
    collection_name: str = Path(
        ..., 
        pattern=r"^[a-zA-Z0-9_-]{3,63}$",
        description="The name of the collection containing the file."
    ),
    app_name: str = Depends(verify_api_key)
):
    """Surgically remove a specific document from a collection without deleting the whole database."""
    return handle_delete_file(collection_name=collection_name, filename=filename, app_name=app_name)

@router.post("/knowledge_base/compare")
@limiter.limit("5/minute") # Protects the GPU: Max 5 comparisons per minute per user
def compare_documents(request: Request, collection_name: str, file_1: str, file_2: str, app_name: str = Depends(verify_api_key)):
    return handle_compare_documents(collection_name=collection_name, file_1=file_1, file_2=file_2, app_name=app_name)

@router.get("/knowledge_base/{collection_name}/files/{filename}/summary")
@limiter.limit("10/minute") # Protects the GPU: Max 10 summaries per minute per user
def summarize_document(request: Request, collection_name: str, filename: str, app_name: str = Depends(verify_api_key)):
    return handle_summarize_document(collection_name=collection_name, filename=filename, app_name=app_name)