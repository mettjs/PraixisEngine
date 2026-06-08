import asyncio

import asyncpg
from pgvector.asyncpg import register_vector

from src.config import POSTGRES_URL as _POSTGRES_URL, EMBEDDING_DIMS as _EMBEDDING_DIMS
from src.utils.vectordb.constants import (
    CREATE_EXTENSION,
    CREATE_UNACCENT,
    CREATE_SCHEMA,
    CREATE_QUESTIONS_SCHEMA,
    PING,
)
from src.utils.vectordb.embeddings import embed

_pool: asyncpg.Pool | None = None


async def init_db() -> None:
    global _pool

    # Validate embedding dims BEFORE creating the schema so a misconfigured
    # EMBEDDING_DIMS surfaces as a clear startup error instead of a cryptic
    # pgvector size mismatch at first insert or query.
    probe = await asyncio.to_thread(embed, ["probe"])
    actual_dims = len(probe[0])
    if actual_dims != _EMBEDDING_DIMS:
        raise RuntimeError(
            f"EMBEDDING_DIMS={_EMBEDDING_DIMS} does not match the model output "
            f"({actual_dims}). Update EMBEDDING_DIMS or change EMBEDDING_MODEL."
        )

    # The extension must exist before the pool opens connections, because
    # register_vector (called via init=) looks up the vector type OID immediately.
    bootstrap = await asyncpg.connect(dsn=_POSTGRES_URL)
    try:
        await bootstrap.execute(CREATE_EXTENSION)
        await bootstrap.execute(CREATE_UNACCENT)
    finally:
        await bootstrap.close()

    async def _init_conn(conn: asyncpg.Connection) -> None:
        await register_vector(conn)
        # Raise HNSW beam width from the default (40) for better ANN recall.
        # 60 is a good balance for RAG workloads; increase further if recall matters
        # more than latency (e.g. 100), decrease for high-throughput low-latency use.
        await conn.execute("SET hnsw.ef_search = 60")

    _pool = await asyncpg.create_pool(
        dsn=_POSTGRES_URL,
        min_size=2,
        max_size=10,
        init=_init_conn,
    )
    async with _pool.acquire() as conn:
        await conn.execute(CREATE_SCHEMA.format(dims=_EMBEDDING_DIMS))
        # chunk_questions has an FK to chunks, so it must be created after it.
        await conn.execute(CREATE_QUESTIONS_SCHEMA.format(dims=_EMBEDDING_DIMS))


async def close_db() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool not initialized.")
    return _pool


async def ping() -> None:
    """Raises if the database is unreachable."""
    await get_pool().fetchval(PING)
