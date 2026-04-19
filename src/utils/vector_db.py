import asyncio
import os
import chromadb
import uuid
from typing import List, Dict, Any, Set
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(_ROOT, "chroma_data"))
chroma_client = chromadb.PersistentClient(path=_CHROMA_PATH)

_embedding_fn = DefaultEmbeddingFunction()


def _scoped_name(app_name: str, collection_name: str) -> str:
    return f"{app_name}_{collection_name}"


async def list_all_collections(app_name: str) -> List[str]:
    all_collections = await asyncio.to_thread(chroma_client.list_collections)
    prefix = f"{app_name}_"
    return [
        col.name[len(prefix):]
        for col in all_collections
        if col.metadata and col.metadata.get("app") == app_name
    ]


async def list_files_in_collection(collection_name: str, app_name: str) -> List[str]:
    def _run():
        try:
            collection = chroma_client.get_collection(name=_scoped_name(app_name, collection_name))
        except Exception:
            raise ValueError(f"The collection '{collection_name}' does not exist.")
        if not collection.metadata or collection.metadata.get("app") != app_name:
            raise ValueError("Access denied: You do not own this collection.")
        results = collection.get(include=["metadatas"])
        unique_files: Set[str] = set()
        metas = results.get("metadatas")
        if metas:
            for meta_dict in metas:
                if meta_dict and "source" in meta_dict:
                    unique_files.add(str(meta_dict["source"]))
        return list(unique_files)

    return await asyncio.to_thread(_run)


async def delete_collection(collection_name: str, app_name: str) -> bool:
    def _run():
        try:
            scoped = _scoped_name(app_name, collection_name)
            collection = chroma_client.get_collection(name=scoped)
            if not collection.metadata or collection.metadata.get("app") != app_name:
                raise ValueError("Access denied: You do not own this collection.")
            chroma_client.delete_collection(name=scoped)
            return True
        except Exception:
            return False

    return await asyncio.to_thread(_run)


async def delete_file_from_collection(collection_name: str, filename: str, app_name: str) -> bool:
    def _run():
        try:
            collection = chroma_client.get_collection(name=_scoped_name(app_name, collection_name))
        except Exception:
            raise ValueError(f"The collection '{collection_name}' does not exist.")
        if not collection.metadata or collection.metadata.get("app") != app_name:
            raise ValueError("Access denied: You do not own this collection.")
        existing_data = collection.get(where={"source": filename}, include=["metadatas"])
        if not existing_data or not existing_data.get("metadatas"):
            raise ValueError(f"The file '{filename}' was not found in '{collection_name}'.")
        collection.delete(where={"source": filename})
        return True

    return await asyncio.to_thread(_run)


async def add_file_to_rag_db(
    text: str,
    collection_name: str,
    filename: str,
    app_name: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> str:
    def _run():
        collection = chroma_client.get_or_create_collection(
            name=_scoped_name(app_name, collection_name),
            metadata={"app": app_name}
        )
        if not collection.metadata or collection.metadata.get("app") != app_name:
            raise ValueError("Access denied: You do not own this collection.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", r"(?<=\. )", " ", ""],
            is_separator_regex=True
        )

        existing = collection.get(where={"source": filename}, include=["metadatas"])
        if existing and existing.get("metadatas"):
            collection.delete(where={"source": filename})

        chunks = text_splitter.split_text(text)
        ids = [f"{filename}_{uuid.uuid4().hex[:6]}" for _ in chunks]
        metadatas: List[Dict[str, Any]] = [{"source": filename, "app": app_name} for _ in chunks]

        collection.add(documents=chunks, metadatas=metadatas, ids=ids)  # type: ignore[arg-type]
        return collection_name

    return await asyncio.to_thread(_run)


async def query_rag_db(
    collection_name: str,
    app_name: str,
    question: str,
    n_results: int = 5,
    metadata_filter: Dict[str, Any] | None = None,
) -> List[Dict[str, str]]:
    def _run():
        collection = chroma_client.get_collection(name=_scoped_name(app_name, collection_name))
        if not collection.metadata or collection.metadata.get("app") != app_name:
            raise ValueError("Access denied: You do not own this collection.")

        count = collection.count()
        if count == 0:
            return []
        actual_n = min(n_results, count)

        where_clause: Dict[str, Any] | None = None
        if metadata_filter:
            where_clause = {"$and": [{"app": app_name}, metadata_filter]}

        results = collection.query(
            query_texts=[question],
            n_results=actual_n,
            where=where_clause,
        )

        retrieved: List[Dict[str, str]] = []
        docs = results.get("documents")
        metas = results.get("metadatas")
        if docs and docs[0] and metas and metas[0]:
            for i in range(len(docs[0])):
                meta_dict = metas[0][i]
                source = str(meta_dict.get("source", "Unknown")) if meta_dict else "Unknown"
                retrieved.append({"source": source, "text": docs[0][i]})
        return retrieved

    return await asyncio.to_thread(_run)


async def get_full_document_text(collection_name: str, app_name: str, filename: str) -> str:
    def _run():
        try:
            collection = chroma_client.get_collection(name=_scoped_name(app_name, collection_name))
        except Exception:
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        if not collection.metadata or collection.metadata.get("app") != app_name:
            raise ValueError("Access denied: You do not own this collection.")

        results = collection.get(where={"source": filename})
        if not results or not results["documents"]:
            raise ValueError(f"No chunks found for document '{filename}' in this collection.")

        return "\n\n".join(results["documents"])  # type: ignore[arg-type]

    return await asyncio.to_thread(_run)


async def get_embedding(text: str) -> list[float]:
    """Returns the embedding vector for a given text using ChromaDB's default embedding function."""
    result = await asyncio.to_thread(_embedding_fn, [text])
    return [float(v) for v in result[0]]
