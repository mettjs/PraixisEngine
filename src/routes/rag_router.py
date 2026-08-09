from fastapi import APIRouter, Depends, File, Path, Query, UploadFile, Form
from src.dependencies.security import caller, verify_api_key
from src.models.schemas import CompareRequest, EmbedRequest, QuestionRequest, SearchRequest, TextUploadRequest
from src.controllers.rag_controller import (
    handle_rag_upload,
    handle_compare_documents,
    handle_delete_file,
    handle_embed,
    handle_file_chunks,
    handle_list_files,
    handle_question_status,
    handle_rag_question,
    handle_regenerate_questions,
    handle_upload_text,
    handle_vector_search,
    handle_list_collections,
    handle_delete_collection,
    handle_summarize_document,
)
from src.utils.system.limiter import rate_limit

router = APIRouter(
    prefix="/rag-db",
    tags=["Vector optimized Endpoints"],
    dependencies=[Depends(verify_api_key)]
)


@router.post("/upload", dependencies=[Depends(rate_limit("15/minute"))])
async def rag_upload_endpoint(
    files: list[UploadFile] = File(..., description="One or more PDF, DOCX, or TXT files — max 20 MB each. "
                                                    "Type is detected from the filename extension, falling back to the "
                                                    "part's Content-Type, then to magic bytes."),
    collection_name: str = Form(default="main", pattern=r"^[a-zA-Z0-9_-]{3,63}$",
                                description="Target collection name. Defaults to 'main'."),
    chunk_size: int = Form(default=2000, ge=100, le=4000, description="Maximum characters per semantic chunk (100–4000)."),
    chunk_overlap: int = Form(default=150, ge=0, le=500, description="Overlap characters between chunks. Only applies when chunking_strategy is 'character'."),
    chunking_strategy: str = Form(default="semantic", pattern=r"^(semantic|character)$",
                                  description="Chunking strategy: 'semantic' cuts at topic shifts using embeddings; 'character' uses fixed-size splits."),
    improved_search: bool = Form(
        default=False,
        description=(
            "Enable hypothetical-question indexing for better natural-language search on this document. "
            "Questions are generated in the background after upload (document is searchable immediately; "
            "matching improves once generation finishes). While running, it occupies one or more GPU slots "
            "from the reserved question-generation pool. Results vary but accuracy usually improves."
        ),
    ),
    app_name: str = Depends(verify_api_key)
):
    return await handle_rag_upload(
        collection_name=collection_name,
        files=files,
        app_name=app_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunking_strategy=chunking_strategy,
        improved_search=improved_search,
    )


@router.post("/upload_text", dependencies=[Depends(rate_limit("15/minute"))])
async def rag_upload_text_endpoint(
    upload_request: TextUploadRequest,
    app_name: str = Depends(verify_api_key)
):
    """Ingest raw text as a document — same pipeline as /upload without the
    multipart step, for callers that already hold the text."""
    return await handle_upload_text(upload_request, app_name=app_name)


@router.post("/ask", dependencies=[Depends(rate_limit("30/minute"))])
async def rag_ask_endpoint(
    question_request: QuestionRequest,
    app_name: str = Depends(verify_api_key),
    caller_entry: dict = Depends(caller),
):
    return await handle_rag_question(question_request, app_name=app_name, caller_entry=caller_entry)


@router.post("/search", dependencies=[Depends(rate_limit("30/minute"))])
async def rag_search_endpoint(
    search_request: SearchRequest,
    app_name: str = Depends(verify_api_key)
):
    return await handle_vector_search(search_request, app_name=app_name)


@router.post("/embed", dependencies=[Depends(rate_limit("60/minute"))])
async def embed_endpoint(
    embed_request: EmbedRequest,
):
    return await handle_embed(embed_request)


@router.get("/list", dependencies=[Depends(rate_limit("60/minute"))])
async def rag_list_endpoint(app_name: str = Depends(verify_api_key)):
    return await handle_list_collections(app_name=app_name)


@router.get("/{collection_name}/files", dependencies=[Depends(rate_limit("60/minute"))])
async def rag_list_files_endpoint(
    collection_name: str = Path(..., pattern=r"^[a-zA-Z0-9_-]{3,63}$"),
    app_name: str = Depends(verify_api_key)
):
    return await handle_list_files(collection_name=collection_name, app_name=app_name)


@router.get("/{collection_name}/files/{filename}/chunks", dependencies=[Depends(rate_limit("30/minute"))])
async def rag_file_chunks_endpoint(
    filename: str = Path(..., description="Exact filename whose stored chunks to return."),
    collection_name: str = Path(..., pattern=r"^[a-zA-Z0-9_-]{3,63}$"),
    app_name: str = Depends(verify_api_key)
):
    """Inspect how a document was chunked — useful for debugging retrieval."""
    return await handle_file_chunks(collection_name=collection_name, filename=filename, app_name=app_name)


@router.post("/{collection_name}/files/{filename}/questions", dependencies=[Depends(rate_limit("10/minute"))])
async def rag_regenerate_questions_endpoint(
    filename: str = Path(..., description="Exact filename to (re)index with hypothetical questions."),
    collection_name: str = Path(..., pattern=r"^[a-zA-Z0-9_-]{3,63}$"),
    app_name: str = Depends(verify_api_key)
):
    """Backfill or rebuild the hypothetical-question index for an already-stored
    file. Existing questions are purged first; generation runs in the background
    (poll the GET endpoint for progress). 409 while a pass is already running."""
    return await handle_regenerate_questions(collection_name=collection_name, filename=filename, app_name=app_name)


@router.get("/{collection_name}/files/{filename}/questions", dependencies=[Depends(rate_limit("60/minute"))])
async def rag_question_status_endpoint(
    filename: str = Path(..., description="Exact filename to report question-index status for."),
    collection_name: str = Path(..., pattern=r"^[a-zA-Z0-9_-]{3,63}$"),
    app_name: str = Depends(verify_api_key)
):
    """Question-index status for a file: chunks stored, questions stored, and
    whether a background generation pass is currently running."""
    return await handle_question_status(collection_name=collection_name, filename=filename, app_name=app_name)


@router.delete("/delete/{collection_name}", dependencies=[Depends(rate_limit("20/minute"))])
async def rag_delete_endpoint(
    collection_name: str = Path(..., pattern=r"^[a-zA-Z0-9_-]{3,63}$"),
    app_name: str = Depends(verify_api_key)
):
    return await handle_delete_collection(collection_name=collection_name, app_name=app_name)


@router.delete("/{collection_name}/files/{filename}", dependencies=[Depends(rate_limit("20/minute"))])
async def rag_delete_file_endpoint(
    filename: str = Path(..., description="Exact filename to delete (e.g. policy_2024.pdf)."),
    collection_name: str = Path(..., pattern=r"^[a-zA-Z0-9_-]{3,63}$"),
    app_name: str = Depends(verify_api_key)
):
    return await handle_delete_file(collection_name=collection_name, filename=filename, app_name=app_name)


@router.post("/knowledge_base/compare", dependencies=[Depends(rate_limit("5/minute"))])
async def rag_compare_documents(
    compare_request: CompareRequest,
    app_name: str = Depends(verify_api_key),
    caller_entry: dict = Depends(caller),
):
    return await handle_compare_documents(
        collection_name=compare_request.collection_name,
        file_1=compare_request.file_1,
        file_2=compare_request.file_2,
        app_name=app_name,
        model=compare_request.model,
        caller_entry=caller_entry,
        stream=compare_request.stream,
        response_format=compare_request.response_format,
    )


@router.get("/knowledge_base/{collection_name}/files/{filename}/summary", dependencies=[Depends(rate_limit("10/minute"))])
async def rag_summarize_document(
    collection_name: str,
    filename: str,
    model: str | None = Query(default=None, pattern=r"^[a-zA-Z0-9_-]{1,63}$", description="Registry id of the LLM to use; omit for this key's default."),
    stream: bool = Query(default=False, description="Stream tokens as text/event-stream, or return one buffered JSON body."),
    response_format: str = Query(default="text", pattern=r"^(text|json)$", description="LLM content format: 'text' or 'json'."),
    app_name: str = Depends(verify_api_key),
    caller_entry: dict = Depends(caller),
):
    return await handle_summarize_document(
        collection_name=collection_name, filename=filename, app_name=app_name,
        model=model, caller_entry=caller_entry, stream=stream, response_format=response_format,
    )
