import os
import chromadb
import uuid
from typing import List, Dict, Any, Set
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(_ROOT, "chroma_data"))
chroma_client = chromadb.PersistentClient(path=_CHROMA_PATH)

def _scoped_name(app_name: str, collection_name: str) -> str:
    """Returns the internal ChromaDB collection name, scoped to the owning app."""
    return f"{app_name}_{collection_name}"

def list_all_collections(app_name: str) -> List[str]:
    all_collections = chroma_client.list_collections()
    prefix = f"{app_name}_"

    collections = [
        col.name[len(prefix):]  # strip the app prefix before returning to the caller
        for col in all_collections
        if col.metadata and col.metadata.get("app") == app_name
    ]

    return collections

def list_files_in_collection(collection_name: str, app_name: str) -> List[str]:
    try:
        collection = chroma_client.get_collection(name=_scoped_name(app_name, collection_name))
    except Exception:
        raise ValueError(f"The collection '{collection_name}' does not exist.")
    
    if not collection.metadata or collection.metadata.get("app") != app_name:
        raise ValueError(f"Access denied: You do not own this collection.")

    # Ask Chroma to return ONLY the metadata (saves memory by not loading the text/vectors)
    results = collection.get(include=["metadatas"])
    
    unique_files: Set[str] = set()
    metas = results.get("metadatas")
    
    if metas:
        for meta_dict in metas:
            if meta_dict and "source" in meta_dict:
                unique_files.add(str(meta_dict["source"]))
                
    return list(unique_files)

def delete_collection(collection_name: str, app_name: str) -> bool:
    try:
        scoped = _scoped_name(app_name, collection_name)
        collection = chroma_client.get_collection(name=scoped)
        if not collection.metadata or collection.metadata.get("app") != app_name:
            raise ValueError(f"Access denied: You do not own this collection.")
        chroma_client.delete_collection(name=scoped)
        return True
    except Exception:
        return False
    
def delete_file_from_collection(collection_name: str, filename: str, app_name: str) -> bool:
    """Deletes all vector chunks associated with a specific file in a collection."""
    try:
        collection = chroma_client.get_collection(name=_scoped_name(app_name, collection_name))
    except Exception:
        raise ValueError(f"The collection '{collection_name}' does not exist.")
    if not collection.metadata or collection.metadata.get("app") != app_name:
        raise ValueError(f"Access denied: You do not own this collection.")
        
    # Peek inside to see if the file actually exists in this collection
    existing_data = collection.get(where={"source": filename}, include=["metadatas"])
    
    # If the list of metadatas is empty, the file isn't there
    if not existing_data or not existing_data.get("metadatas"):
        raise ValueError(f"The file '{filename}' was not found in '{collection_name}'.")
        
    # Perform the targeted deletion using the metadata filter
    collection.delete(where={"source": filename})
    
    return True

def add_file_to_rag_db(text: str, collection_name: str, filename: str, app_name: str) -> str:
    """Chunks the text by paragraphs and sentences, then stores it."""

    collection = chroma_client.get_or_create_collection(
        name=_scoped_name(app_name, collection_name),
        metadata={"app": app_name}
        )
    
    if not collection.metadata or collection.metadata.get("app") != app_name:
        raise ValueError(f"Access denied: You do not own this collection.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        # It attempts to split on the first separator. If the chunk is still too big, 
        # it falls back to the next one, ensuring paragraphs/sentences aren't cut in half.
        separators=["\n\n", "\n", r"(?<=\. )", " ", ""],
        is_separator_regex=True
    )
    
    # If this file was uploaded before, remove the old chunks first to prevent duplicates
    existing = collection.get(where={"source": filename}, include=["metadatas"])
    if existing and existing.get("metadatas"):
        collection.delete(where={"source": filename})

    chunks = text_splitter.split_text(text)

    ids = [f"{filename}_{uuid.uuid4().hex[:6]}" for _ in chunks]
    metadatas: List[Dict[str, Any]] = [{"source": filename, "app": app_name} for _ in chunks]
    
    collection.add(
        documents=chunks, 
        metadatas=metadatas,  # type: ignore[arg-type]
        ids=ids
    )
    
    return collection_name

def query_rag_db(collection_name: str, app_name: str, question: str, n_results: int = 5) -> List[Dict[str, str]]:
    """Searches a specific collection and returns the text and source."""

    collection = chroma_client.get_collection(name=_scoped_name(app_name, collection_name))
    if not collection.metadata or collection.metadata.get("app") != app_name:
        raise ValueError(f"Access denied: You do not own this collection.")

    # ChromaDB raises if n_results > number of stored vectors
    count = collection.count()
    if count == 0:
        return []
    n_results = min(n_results, count)

    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    
    retrieved_data: List[Dict[str, str]] = []
    
    docs = results.get("documents")
    metas = results.get("metadatas")
    
    if docs and docs[0] and metas and metas[0]:
        for i in range(len(docs[0])):
            chunk_text = docs[0][i]
            meta_dict = metas[0][i]
            source_file = str(meta_dict.get("source", "Unknown")) if meta_dict else "Unknown"
            
            retrieved_data.append({
                "source": source_file,
                "text": chunk_text
            })
            
    return retrieved_data

def get_full_document_text(collection_name: str, app_name: str, filename: str) -> str:
    """Retrieves all chunks of a specific document from ChromaDB and stitches them together."""
    try:
        collection = chroma_client.get_collection(name=_scoped_name(app_name, collection_name))
    except Exception:
        raise ValueError(f"Collection '{collection_name}' does not exist.")
    if not collection.metadata or collection.metadata.get("app") != app_name:
        raise ValueError(f"Access denied: You do not own this collection.")
        
    results = collection.get(where={"source": filename})
    
    if not results or not results["documents"]:
        raise ValueError(f"No chunks found for document '{filename}' in this collection.")
        
    document_text = "\n\n".join(results["documents"]) # type: ignore[arg-type]
    
    return document_text