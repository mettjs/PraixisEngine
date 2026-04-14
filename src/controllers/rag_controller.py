from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from src.models.schemas import QuestionRequest
from src.services.rag_service import generate_comparison, generate_rag_answer, generate_summary, reformulate_query
from src.utils.file_parser import extract_text_from_file
from src.utils.vector_db import (
    delete_file_from_collection,
    list_all_collections,
    delete_collection, 
    add_file_to_rag_db,
    list_files_in_collection, 
    query_rag_db,
    get_full_document_text
    )
from src.utils.memory import get_session_history
from src.utils.logger import logger
from src.utils.concurrency import GPUBusyError, acquire_gpu_slot, release_gpu_slot

_MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

def handle_list_collections(app_name: str) -> dict:
    """Gets all collections and packages them into a clean JSON response."""
    try:
        collections = list_all_collections(app_name=app_name)
        logger.info(f"Listed collections for app: {app_name}, total_collections: {len(collections)}")
        return {
            "status": "success",
            "total_documents": len(collections),
            "active_collections": collections
        }
    except Exception as e:
        logger.error(f"Error in handle_list_collections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
def handle_list_files(collection_name: str, app_name: str) -> dict:
    """Gets all unique files in a collection and packages the JSON response."""
    try:
        files = list_files_in_collection(collection_name=collection_name, app_name=app_name)
        logger.info(f"Listed files in collection: {collection_name} for app: {app_name}, total_files: {len(files)}")
        return {
            "status": "success",
            "collection_name": collection_name,
            "total_files": len(files),
            "files_stored": files
        }
    except ValueError as ve:
        logger.warning(f"Value error in handle_list_files: {str(ve)}")
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in handle_list_files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def handle_delete_collection(collection_name: str, app_name: str) -> dict:
    """Attempts to delete a collection and returns a 404 error if it fails."""
    success = delete_collection(collection_name=collection_name, app_name=app_name)
    
    if not success:
        logger.warning(f"Collection not found for deletion for app: {app_name}, collection: {collection_name}")
        raise HTTPException(
            status_code=404, 
            detail=f"Error: Collection '{collection_name}' does not exist or was already deleted."
        )
    logger.info(f"Deleted collection: {collection_name} for app: {app_name}")
    return {
        "status": "success",
        "message": f"The collection '{collection_name}' has been permanently deleted."
    }
    
def handle_delete_file(collection_name: str, filename: str, app_name: str) -> dict:
    """Attempts to delete a specific file and packages the JSON response."""
    try:
        delete_file_from_collection(collection_name=collection_name, filename=filename, app_name=app_name)
        logger.info(f"Deleted file: {filename} from collection: {collection_name} for app: {app_name}")
        return {
            "status": "success",
            "message": f"All data for '{filename}' has been permanently removed from '{collection_name}'."
        }
    except ValueError as ve:
        logger.warning(f"Value error in handle_delete_file: {str(ve)}")
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in handle_delete_file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def handle_rag_upload(collection_name: str, file: UploadFile, app_name: str) -> dict:
    if not file.filename:
        logger.warning(f"RAG upload request received without a file for app: {app_name}")
        raise HTTPException(status_code=400, detail="No file uploaded.")
    try:
        logger.info(f"Uploading file: {file.filename} to collection: {collection_name} for app: {app_name}")
        content = await file.read(_MAX_FILE_SIZE + 1)
        if len(content) > _MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum allowed size is 20 MB.")

        document_text = extract_text_from_file(file.filename, content)
        if not document_text.strip():
            logger.warning(f"Could not extract any text from the provided file: {file.filename} for app: {app_name}")
            raise ValueError("File is empty or unreadable.")
            
        collection_id = add_file_to_rag_db(
            text=document_text, 
            collection_name=collection_name, 
            filename=file.filename, 
            app_name=app_name)
        logger.info(f"Uploaded file: {file.filename} to collection: {collection_id} for app: {app_name}")
        return {
            "status": "success",
            "message": "Document indexed successfully.",
            "collection_name": collection_id 
        }
    except HTTPException:
        raise
    except ValueError as ve:
        logger.warning(f"Value error in handle_rag_upload: {str(ve)}")
        raise HTTPException(status_code=400, detail="Invalid file format.")
    except Exception as e:
        logger.error(f"Error in handle_rag_upload: {str(e)}")
        raise HTTPException(status_code=500, detail="RAG Upload Error")

def handle_rag_question(request: QuestionRequest, app_name: str) -> StreamingResponse:
    try:
        # Fetch existing history for query reformulation
        history = get_session_history(session_id=request.session_id, app_name=app_name) if request.session_id else []

        # Reformulate vague follow-up questions into standalone search queries.
        # This makes a GPU call internally — GPUBusyError propagates up if the GPU is full.
        search_query = reformulate_query(history, request.question)

        relevant_chunks = query_rag_db(
            collection_name=request.collection_name,
            app_name=app_name,
            question=search_query,
            n_results=request.n_results
        )
    except GPUBusyError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Error preparing RAG question for app: {app_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG Generation Error")

    try:
        acquire_gpu_slot()
    except GPUBusyError as e:
        raise HTTPException(status_code=503, detail="Server is too busy. Please try again later.")

    def _stream():
        try:
            yield from generate_rag_answer(
                question=request.question,
                app_name=app_name,
                context_chunks=relevant_chunks,
                search_query=search_query,
                session_id=request.session_id,
                system_prompt=request.system_prompt
            )
        finally:
            release_gpu_slot()

    try:
        logger.info(f"Streaming RAG answer for app: {app_name}, collection: {request.collection_name}")
        return StreamingResponse(_stream(), media_type="text/event-stream")
    except Exception as e:
        release_gpu_slot()
        logger.error(f"Error in handle_rag_question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Rag answer generation error")
    
def handle_summarize_document(collection_name: str, filename: str, app_name: str) -> dict:
    try:
        document_text = get_full_document_text(collection_name=collection_name, app_name=app_name, filename=filename)
        summary = generate_summary(document_text)
        logger.info(f"Generated {filename} summary for app: {app_name}")
        return {"filename": filename, "summary": summary}
    except GPUBusyError as e:
        raise HTTPException(status_code=503, detail="Server is too busy. Please try again later.")
    except Exception as e:
        logger.error(f"Error in handle_summarize_document: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the summary.")

def handle_compare_documents(collection_name: str, file_1: str, file_2: str, app_name: str) -> dict:
    try:
        doc1_text = get_full_document_text(collection_name=collection_name, app_name=app_name, filename=file_1)
        doc2_text = get_full_document_text(collection_name=collection_name, app_name=app_name, filename=file_2)
        comparison = generate_comparison(doc1_text, doc2_text, file_1, file_2)
        logger.info(f"Generated comparison between {file_1} and {file_2} for app: {app_name}")
        return {"file_1": file_1, "file_2": file_2, "comparison": comparison}
    except GPUBusyError as e:
        raise HTTPException(status_code=503, detail="Server is too busy. Please try again later.")
    except Exception as e:
        logger.error(f"Error in handle_compare_documents: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the comparison.")