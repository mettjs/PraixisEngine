import asyncio
from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from src.models.schemas import EmbedRequest, QuestionRequest
from src.services.rag_service import generate_comparison, generate_rag_answer, generate_summary, reformulate_query
from src.utils.file_parser import extract_text_from_file, MAX_FILE_SIZE
from src.utils.vectordb.embeddings import get_embedding
from src.utils.vectordb.ingestion import add_file_to_rag_db
from src.utils.vectordb.questions import schedule_question_generation
from src.utils.vectordb.collections import (
    collection_exists,
    list_all_collections,
    list_files_in_collection,
    delete_collection,
    delete_file_from_collection,
)
from src.utils.vectordb.retrieval import query_rag_db, get_full_document_text
from src.utils.store.sessions import get_session_history
from src.utils.system.logger import logger
from src.utils.system.streaming import SlotReleasingStreamingResponse, drain_to_json
from src.utils.concurrency import GPUBusyError, acquire_gpu_slot
from src.utils.store.audit import log_event


async def handle_list_collections(app_name: str) -> dict:
    try:
        collections = await list_all_collections(app_name=app_name)
        logger.info(f"Listed collections for app: {app_name}, total_collections: {len(collections)}")
        return {"status": "success", "total_documents": len(collections), "active_collections": collections}
    except Exception as e:
        logger.error(f"Error in handle_list_collections: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal error.")


async def handle_list_files(collection_name: str, app_name: str) -> dict:
    try:
        files = await list_files_in_collection(collection_name=collection_name, app_name=app_name)
        logger.info(f"Listed files in collection: {collection_name} for app: {app_name}, total_files: {len(files)}")
        return {"status": "success", "collection_name": collection_name, "total_files": len(files), "files_stored": files}
    except ValueError as ve:
        logger.warning(f"Value error in handle_list_files: {str(ve)}")
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in handle_list_files: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal error.")


async def handle_delete_collection(collection_name: str, app_name: str) -> dict:
    success = await delete_collection(collection_name=collection_name, app_name=app_name)
    if not success:
        logger.warning(f"Collection not found for deletion for app: {app_name}, collection: {collection_name}")
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' does not exist or was already deleted.")
    logger.info(f"Deleted collection: {collection_name} for app: {app_name}")
    await log_event("COLLECTION_DELETED", {"collection": collection_name}, app_name=app_name)
    return {"status": "success", "message": f"The collection '{collection_name}' has been permanently deleted."}


async def handle_delete_file(collection_name: str, filename: str, app_name: str) -> dict:
    try:
        await delete_file_from_collection(collection_name=collection_name, filename=filename, app_name=app_name)
        logger.info(f"Deleted file: {filename} from collection: {collection_name} for app: {app_name}")
        await log_event("FILE_DELETED", {"filename": filename, "collection": collection_name}, app_name=app_name)
        return {"status": "success", "message": f"All data for '{filename}' has been permanently removed from '{collection_name}'."}
    except ValueError as ve:
        logger.warning(f"Value error in handle_delete_file: {str(ve)}")
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in handle_delete_file: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal error.")

async def handle_rag_upload(
    collection_name: str,
    files: list[UploadFile],
    app_name: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 150,
    chunking_strategy: str = "semantic",
    improved_search: bool = False,
) -> dict:
    if chunking_strategy not in ("semantic", "character"):
        raise HTTPException(status_code=422, detail="chunking_strategy must be 'semantic' or 'character'.")
    if chunking_strategy == "character" and chunk_overlap >= chunk_size:
        raise HTTPException(status_code=422, detail="chunk_overlap must be less than chunk_size.")
    results = []
    for file in files:
        if not file.filename:
            results.append({"filename": None, "status": "error", "detail": "File has no filename."})
            continue
        try:
            content = await file.read(MAX_FILE_SIZE + 1)
            if len(content) > MAX_FILE_SIZE:
                results.append({"filename": file.filename, "status": "error", "detail": "File exceeds 20 MB limit."})
                continue
            document_text = extract_text_from_file(file.filename, content, content_type=file.content_type)
            if not document_text.strip():
                results.append({"filename": file.filename, "status": "error", "detail": "File is empty or unreadable."})
                continue
            chunk_rows = await add_file_to_rag_db(
                text=document_text,
                collection_name=collection_name,
                filename=file.filename,
                app_name=app_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunking_strategy=chunking_strategy,
            )
            # Chunks are now searchable. When improved_search is requested,
            # generate hypothetical questions in the background so the upload
            # response is not blocked on LLM generation.
            if improved_search:
                schedule_question_generation(
                    app_name=app_name,
                    collection_name=collection_name,
                    source=file.filename,
                    chunks=chunk_rows,
                )
            logger.info(f"Batch uploaded file: {file.filename} to collection: {collection_name} for app: {app_name}")
            await log_event("FILE_UPLOADED", {"filename": file.filename, "collection": collection_name}, app_name=app_name)
            results.append({"filename": file.filename, "status": "success"})
        except ValueError as ve:
            results.append({"filename": file.filename, "status": "error", "detail": str(ve)})
        except Exception as e:
            logger.error(f"Batch upload error for {file.filename}: {e}")
            results.append({"filename": file.filename, "status": "error", "detail": "Internal error."})
    success_count = sum(1 for r in results if r["status"] == "success")
    return {"collection_name": collection_name, "processed": len(results), "succeeded": success_count, "results": results}


async def handle_rag_question(request: QuestionRequest, app_name: str) -> StreamingResponse:
    # Check the collection before any LLM work, so a typo'd name is a clean 404
    # instead of a burned reformulation call and a streamed non-answer.
    if not await collection_exists(collection_name=request.collection_name, app_name=app_name):
        raise HTTPException(status_code=404, detail=f"Collection '{request.collection_name}' does not exist.")

    try:
        history = await get_session_history(session_id=request.session_id, app_name=app_name) if request.session_id else []
        search_query = await reformulate_query(history, request.question, app_name=app_name)
        relevant_chunks = await query_rag_db(
            collection_name=request.collection_name,
            app_name=app_name,
            question=search_query,
            n_results=request.n_results,
            metadata_filter=request.metadata_filter,
        )
    except GPUBusyError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Error preparing RAG question for app: {app_name}: {str(e)}")
        raise HTTPException(status_code=500, detail="RAG Generation Error")

    # The collection exists, so an empty result means the metadata filter
    # excluded everything. Answering would just stream "not found in context"
    # while holding a GPU slot — return early instead.
    if not relevant_chunks:
        raise HTTPException(
            status_code=404,
            detail="No content matched the question in this collection. Check the metadata filter.",
        )

    try:
        slot = await acquire_gpu_slot()
    except GPUBusyError as e:
        raise HTTPException(status_code=503, detail=str(e))

    answer = generate_rag_answer(
        question=request.question,
        app_name=app_name,
        context_chunks=relevant_chunks,
        search_query=search_query,
        session_id=request.session_id,
        system_prompt=request.system_prompt,
        response_format=request.response_format,
    )

    # The streaming wrapper owns the slot and releases it whether the stream
    # completes, errors, or the client disconnects before it even starts. The
    # buffered path drains the same generator and must release the slot itself.
    # If anything fails before either takes over, release here.
    try:
        if request.stream:
            logger.info(f"Streaming RAG answer for app: {app_name}, collection: {request.collection_name}")
            return SlotReleasingStreamingResponse(answer, slot=slot, media_type="text/event-stream")
        logger.info(f"Buffering RAG answer for app: {app_name}, collection: {request.collection_name}")
        try:
            return await drain_to_json(answer)
        finally:
            await slot.release()
    except Exception:
        await slot.release()
        raise


async def handle_summarize_document(
    collection_name: str, filename: str, app_name: str, stream: bool = False, response_format: str = "text"
) -> StreamingResponse | dict:
    try:
        document_text = await get_full_document_text(collection_name=collection_name, app_name=app_name, filename=filename)
    except GPUBusyError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Error in handle_summarize_document: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the summary.")

    # generate_summary self-acquires a GPU slot per LLM call, so we must NOT
    # pre-acquire here.
    async def _summary_with_header():
        yield f"[FILE:{filename}]\n"
        async for piece in generate_summary(document_text, app_name=app_name, response_format=response_format):
            yield piece

    if not stream:
        try:
            body = await drain_to_json(_summary_with_header())
        except GPUBusyError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.error(f"Error in handle_summarize_document: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while generating the summary.")
        logger.info(f"Generated {filename} summary for app: {app_name}")
        return body

    async def _guarded():
        try:
            async for piece in _summary_with_header():
                yield piece
        except GPUBusyError as e:
            yield f"[ERROR:{e}]\n"

    logger.info(f"Streaming {filename} summary for app: {app_name}")
    return StreamingResponse(_guarded(), media_type="text/event-stream")


async def handle_compare_documents(
    collection_name: str, file_1: str, file_2: str, app_name: str, stream: bool = False, response_format: str = "text"
) -> StreamingResponse | dict:
    try:
        doc1_text, doc2_text = await asyncio.gather(
            get_full_document_text(collection_name=collection_name, app_name=app_name, filename=file_1),
            get_full_document_text(collection_name=collection_name, app_name=app_name, filename=file_2),
        )
    except GPUBusyError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Error in handle_compare_documents: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the comparison.")

    # generate_comparison self-acquires a GPU slot per LLM call, so we must NOT
    # pre-acquire here.
    comparison = generate_comparison(doc1_text, doc2_text, file_1, file_2, app_name=app_name, response_format=response_format)

    if not stream:
        try:
            body = await drain_to_json(comparison)
        except GPUBusyError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.error(f"Error in handle_compare_documents: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while generating the comparison.")
        logger.info(f"Generated comparison between {file_1} and {file_2} for app: {app_name}")
        # Echo the request's filenames alongside the generated content.
        return {"file_1": file_1, "file_2": file_2, **body}

    async def _guarded():
        try:
            async for piece in comparison:
                yield piece
        except GPUBusyError as e:
            yield f"[ERROR:{e}]\n"

    logger.info(f"Streaming comparison between {file_1} and {file_2} for app: {app_name}")
    return StreamingResponse(_guarded(), media_type="text/event-stream")


async def handle_embed(request: EmbedRequest) -> dict:
    try:
        vector = await get_embedding(request.text)
        return {"text": request.text, "dimensions": len(vector), "embedding": vector}
    except Exception as e:
        logger.error(f"Embed error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding.")
