"""Vector store factory.

The backend is chosen once per process from ``VECTOR_BACKEND`` and exposed as
a singleton; everything outside this package depends only on the
:class:`~src.utils.vectordb.base.VectorStore` interface. Backend packages are
imported lazily so selecting one never pays the other's import cost.
"""
from src.config import VECTOR_BACKEND as _VECTOR_BACKEND
from src.utils.vectordb.base import VectorStore

_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        if _VECTOR_BACKEND == "pgvector":
            from src.utils.vectordb.pg.store import PgVectorStore
            _store = PgVectorStore()
        elif _VECTOR_BACKEND == "chroma":
            from src.utils.vectordb.chroma.store import ChromaStore
            _store = ChromaStore()
        else:
            raise RuntimeError(
                f"Unknown VECTOR_BACKEND '{_VECTOR_BACKEND}'. Use 'pgvector' or 'chroma'."
            )
    return _store
