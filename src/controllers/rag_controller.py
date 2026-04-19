from typing import List
from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from src.models.schemas import EmbedRequest, QuestionRequest
from src.services.rag_service import generate_comparison, generate_rag_answer, generate_summary, reformulate_query
from src.utils.file_parser import extract_text_from_file
from src.utils.vector_db import (
    delete_file_from_collection,
    get_embedding,
    list_all_collections,
    delete_collection,
    add_file_to_rag_db,
    list_files_in_collection,
    query_rag_db,
    get_full_document_text,
)
from src.utils.memory import get_session_history
from src.utils.logger import logger
from src.utils.concurrency import GPUBusyError, acquire_gpu_slot, release_gpu_slot

_MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


async def handle_list_collections(app_name: str) -> dict:
    try:
        collections = await list_all_collections(app_name=app_name)
        logger.info(f"Listed collections for app: {app_name}, total_collections: {len(collections)}")
        return {"status": "success", "total_documents": len(collections), "active_collections": collections}
    except Exception as e:
        logger.error(f"Error in handle_list_collections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


async def handle_delete_collection(collection_name: str, app_name: str) -> dict:
    success = await delete_collection(collection_name=collection_name, app_name=app_name)
    if not success:
        logger.warning(f"Collection not found for deletion for app: {app_name}, collection: {collection_name}")
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' does not exist or was already deleted.")
    logger.info(f"Deleted collection: {collection_name} for app: {app_name}")
    return {"status": "success", "message": f"The collection '{collection_name}' has been permanently deleted."}


async def handle_delete_file(collection_name: str, filename: str, app_name: str) -> dict:
    try:
        await delete_file_from_collection(collection_name=collection_name, filename=filename, app_name=app_name)
        logger.info(f"Deleted file: {filename} from collection: {collection_name} for app: {app_name}")
        return {"status": "success", "message": f"All data for '{filename}' has been permanently removed from '{collection_name}'."}
    except ValueError as ve:
        logger.warning(f"Value error in handle_delete_file: {str(ve)}")
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in handle_delete_file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def handle_rag_upload(
    collection_name: str,
    files: List[UploadFile],
    app_name: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> dict:
    results = []
    for file in files:
        if not file.filename:
            results.append({"filename": None, "status": "error", "detail": "File has no filename."})
            continue
        try:
            content = await file.read(_MAX_FILE_SIZE + 1)
            if len(content) > _MAX_FILE_SIZE:
                results.append({"filename": file.filename, "status": "error", "detail": "File exceeds 20 MB limit."})
                continue
            document_text = extract_text_from_file(file.filename, content)
            if not document_text.strip():
                results.append({"filename": file.filename, "status": "error", "detail": "File is empty or unreadable."})
                continue
            await add_file_to_rag_db(
                text=document_text,
                collection_name=collection_name,
                filename=file.filename,
                app_name=app_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            logger.info(f"Batch uploaded file: {file.filename} to collection: {collection_name} for app: {app_name}")
            results.append({"filename": file.filename, "status": "success"})
        except ValueError as ve:
            results.append({"filename": file.filename, "status": "error", "detail": str(ve)})
        except Exception as e:
            logger.error(f"Batch upload error for {file.filename}: {e}")
            results.append({"filename": file.filename, "status": "error", "detail": "Internal error."})
    success_count = sum(1 for r in results if r["status"] == "success")
    return {"collection_name": collection_name, "processed": len(results), "succeeded": success_count, "results": results}


async def handle_rag_question(request: QuestionRequest, app_name: str) -> StreamingResponse:
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

    try:
        await acquire_gpu_slot()
    except GPUBusyError:
        raise HTTPException(status_code=503, detail="Server is too busy. Please try again later.")

    async def _stream():
        try:
            async for token in generate_rag_answer(
                question=request.question,
                app_name=app_name,
                context_chunks=relevant_chunks,
                search_query=search_query,
                session_id=request.session_id,
                system_prompt=request.system_prompt,
            ):
                yield token
        finally:
            await release_gpu_slot()

    try:
        logger.info(f"Streaming RAG answer for app: {app_name}, collection: {request.collection_name}")
        return StreamingResponse(_stream(), media_type="text/event-stream")
    except Exception as e:
        await release_gpu_slot()
        logger.error(f"Error in handle_rag_question: {str(e)}")
        raise HTTPException(status_code=500, detail="RAG answer generation error")


async def handle_summarize_document(collection_name: str, filename: str, app_name: str) -> dict:
    try:
        document_text = await get_full_document_text(collection_name=collection_name, app_name=app_name, filename=filename)
        summary = await generate_summary(document_text, app_name=app_name)
        logger.info(f"Generated {filename} summary for app: {app_name}")
        return {"filename": filename, "summary": summary}
    except GPUBusyError as e:
        raise HTTPException(status_code=503, detail="Server is too busy. Please try again later.")
    except Exception as e:
        logger.error(f"Error in handle_summarize_document: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the summary.")


async def handle_compare_documents(collection_name: str, file_1: str, file_2: str, app_name: str) -> dict:
    try:
        doc1_text = await get_full_document_text(collection_name=collection_name, app_name=app_name, filename=file_1)
        doc2_text = await get_full_document_text(collection_name=collection_name, app_name=app_name, filename=file_2)
        comparison = await generate_comparison(doc1_text, doc2_text, file_1, file_2, app_name=app_name)
        logger.info(f"Generated comparison between {file_1} and {file_2} for app: {app_name}")
        return {"file_1": file_1, "file_2": file_2, "comparison": comparison}
    except GPUBusyError as e:
        raise HTTPException(status_code=503, detail="Server is too busy. Please try again later.")
    except Exception as e:
        logger.error(f"Error in handle_compare_documents: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the comparison.")


async def handle_embed(request: EmbedRequest) -> dict:
    try:
        vector = await get_embedding(request.text)
        return {"text": request.text, "dimensions": len(vector), "embedding": vector}
    except Exception as e:
        logger.error(f"Embed error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding.")
