import asyncio
import chromadb
import uuid
from typing import List, Dict, Any, Set
from chromadb.utils.embedding_functions import EmbeddingFunction
from chromadb.api.types import Documents, Embeddings
from fastembed import TextEmbedding

from src.config import CHROMA_PATH as _CHROMA_PATH, EMBEDDING_MODEL as _EMBEDDING_MODEL
from src.utils.documents.chunking import character_chunk, semantic_chunk

chroma_client = chromadb.PersistentClient(path=_CHROMA_PATH)


class _FastEmbedFn(EmbeddingFunction[Documents]):  # type: ignore[misc]
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._embedder = TextEmbedding(model_name=model_name)

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        return [emb.tolist() for emb in self._embedder.embed(input)]

    @staticmethod
    def name() -> str:
        return "fastembed"

    def get_config(self) -> dict:
        return {"model_name": self._model_name}

    @staticmethod
    def build_from_config(config: dict) -> "_FastEmbedFn":
        return _FastEmbedFn(model_name=config["model_name"])


_embedding_fn = _FastEmbedFn(model_name=_EMBEDDING_MODEL)


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
            collection = chroma_client.get_collection(name=_scoped_name(app_name, collection_name), embedding_function=_embedding_fn)
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
            collection = chroma_client.get_collection(name=scoped, embedding_function=_embedding_fn)
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
            collection = chroma_client.get_collection(name=_scoped_name(app_name, collection_name), embedding_function=_embedding_fn)
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
    chunk_size: int = 2000,
    chunk_overlap: int = 150,
    chunking_strategy: str = "semantic",
) -> str:
    if chunking_strategy == "semantic":
        chunks = await asyncio.to_thread(semantic_chunk, text, max_chunk_chars=chunk_size)
    elif chunking_strategy == "character":
        chunks = await asyncio.to_thread(character_chunk, text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError("Invalid chunking strategy. Use 'semantic' or 'character'.")

    if not chunks:
        raise ValueError("Document produced no chunks; nothing to ingest.")

    def _run():
        collection = chroma_client.get_or_create_collection(
            name=_scoped_name(app_name, collection_name),
            metadata={"app": app_name},
            embedding_function=_embedding_fn,
        )
        if not collection.metadata or collection.metadata.get("app") != app_name:
            raise ValueError("Access denied: You do not own this collection.")

        existing = collection.get(where={"source": filename}, include=["metadatas"])
        old_ids: List[str] = existing.get("ids", []) if existing else []

        new_ids = [f"{filename}_{uuid.uuid4().hex[:6]}" for _ in chunks]
        metadatas: List[Dict[str, Any]] = [
            {"source": filename, "app": app_name, "chunk_index": i}
            for i in range(len(chunks))
        ]

        # Add new chunks first — if this fails, old data is still intact
        collection.add(documents=chunks, metadatas=metadatas, ids=new_ids)  # type: ignore[arg-type]

        # Remove old chunks only after successful add
        if old_ids:
            collection.delete(ids=old_ids)
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
        collection = chroma_client.get_collection(name=_scoped_name(app_name, collection_name), embedding_function=_embedding_fn)
        if not collection.metadata or collection.metadata.get("app") != app_name:
            raise ValueError("Access denied: You do not own this collection.")

        count = collection.count()
        if count == 0:
            return []

        where_clause: Dict[str, Any] | None = None
        if metadata_filter:
            where_clause = {"$and": [{"app": app_name}, metadata_filter]}

        # When a filter is active, cap n_results against the matching subset, not
        # the total. Passing n_results > filtered_count causes a ChromaDB error.
        if where_clause is not None:
            filtered_ids = collection.get(where=where_clause, include=[]).get("ids") or []
            actual_n = min(n_results, len(filtered_ids))
            if actual_n == 0:
                return []
        else:
            actual_n = min(n_results, count)

        query_kwargs: Dict[str, Any] = {
            "query_texts": [question],
            "n_results": actual_n,
            "include": ["documents", "metadatas"],
        }
        if where_clause is not None:
            query_kwargs["where"] = where_clause

        results = collection.query(**query_kwargs)

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


async def search_collection(
    collection_name: str,
    app_name: str,
    query: str,
    n_results: int = 5,
) -> List[Dict[str, Any]]:
    def _run():
        try:
            collection = chroma_client.get_collection(name=_scoped_name(app_name, collection_name), embedding_function=_embedding_fn)
        except Exception:
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        if not collection.metadata or collection.metadata.get("app") != app_name:
            raise ValueError("Access denied.")
        count = collection.count()
        if count == 0:
            return []
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )
        docs      = results.get("documents")
        metas     = results.get("metadatas")
        distances = results.get("distances")
        retrieved = []
        if docs and docs[0]:
            for i in range(len(docs[0])):
                meta_dict = metas[0][i] if metas and metas[0] else {}
                distance  = distances[0][i] if distances and distances[0] else None
                retrieved.append({
                    "source": str(meta_dict.get("source", "Unknown")) if meta_dict else "Unknown",
                    "text": docs[0][i],
                    "score": round(1 / (1 + distance), 3) if distance is not None else None,
                })
        return retrieved

    return await asyncio.to_thread(_run)


async def get_full_document_text(collection_name: str, app_name: str, filename: str) -> str:
    def _run():
        try:
            collection = chroma_client.get_collection(name=_scoped_name(app_name, collection_name), embedding_function=_embedding_fn)
        except Exception:
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        if not collection.metadata or collection.metadata.get("app") != app_name:
            raise ValueError("Access denied: You do not own this collection.")

        results = collection.get(where={"source": filename}, include=["documents", "metadatas"])
        documents = results.get("documents") if results else None
        if not documents:
            raise ValueError(f"No chunks found for document '{filename}' in this collection.")

        metas = results.get("metadatas")
        if not metas or len(metas) != len(documents):
            # Metadata unavailable or misaligned — fall back to Chroma's order
            # rather than risk dropping chunks via a truncated zip.
            return "\n\n".join(documents)

        # Chroma does not guarantee retrieval order; sort by the stored chunk_index
        # so the reconstructed document preserves its original sequence. Chunks
        # written before chunk_index existed sort last in stable insertion order.
        ordered = sorted(
            zip(documents, metas),
            key=lambda pair: pair[1].get("chunk_index", float("inf")) if pair[1] else float("inf"),
        )
        return "\n\n".join(doc for doc, _ in ordered)

    return await asyncio.to_thread(_run)


async def get_embedding(text: str) -> list[float]:
    """Returns the embedding vector for a given text using ChromaDB's default embedding function."""
    result = await asyncio.to_thread(_embedding_fn, [text])
    return [float(v) for v in result[0]]
