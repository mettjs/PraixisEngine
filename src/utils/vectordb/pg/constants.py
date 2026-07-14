# ── Schema DDL ────────────────────────────────────────────────────────────────

CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"
CREATE_UNACCENT = "CREATE EXTENSION IF NOT EXISTS unaccent"

# Requires .format(dims=EMBEDDING_DIMS) at the call site.
CREATE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS chunks (
        id          TEXT PRIMARY KEY,
        app         TEXT NOT NULL,
        collection  TEXT NOT NULL,
        source      TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        content     TEXT NOT NULL,
        embedding   vector({dims}),
        tsv         tsvector
    );

    CREATE INDEX IF NOT EXISTS chunks_app_col_idx
        ON chunks (app, collection);
    CREATE INDEX IF NOT EXISTS chunks_source_idx
        ON chunks (app, collection, source, chunk_index);
    CREATE INDEX IF NOT EXISTS chunks_fts_idx
        ON chunks USING gin (tsv);
    CREATE INDEX IF NOT EXISTS chunks_hnsw_idx
        ON chunks USING hnsw (embedding vector_cosine_ops);
"""

# Hypothetical-question index: one row per LLM-generated question, each pointing
# back to its parent chunk. Querying these (question-to-question) closes the
# genre gap between civilian phrasing and formal source text.
# chunk_index is denormalized from the parent so window expansion needs no join.
# ON DELETE CASCADE means deleting a file's chunks (DELETE_FILE, or the re-upload
# DELETE->reinsert txn) automatically purges its questions — no extra bookkeeping.
# Requires .format(dims=EMBEDDING_DIMS) at the call site.
CREATE_QUESTIONS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS chunk_questions (
        id          TEXT PRIMARY KEY,
        app         TEXT NOT NULL,
        collection  TEXT NOT NULL,
        source      TEXT NOT NULL,
        chunk_id    TEXT NOT NULL REFERENCES chunks (id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        question    TEXT NOT NULL,
        embedding   vector({dims})
    );

    CREATE INDEX IF NOT EXISTS chunk_questions_app_col_idx
        ON chunk_questions (app, collection);
    CREATE INDEX IF NOT EXISTS chunk_questions_chunk_id_idx
        ON chunk_questions (chunk_id);
    CREATE INDEX IF NOT EXISTS chunk_questions_hnsw_idx
        ON chunk_questions USING hnsw (embedding vector_cosine_ops);
"""

# ── Health ────────────────────────────────────────────────────────────────────

PING = "SELECT 1"

# ── Admin ─────────────────────────────────────────────────────────────────────

ALL_COLLECTIONS_ADMIN = """
    SELECT app, collection, COUNT(*) AS chunk_count
    FROM chunks
    GROUP BY app, collection
    ORDER BY app, collection
"""

VECTOR_STATS = """
    SELECT
        COUNT(DISTINCT app || ':' || collection) AS cols,
        COUNT(*) AS chunks
    FROM chunks
"""

# ── Collections ───────────────────────────────────────────────────────────────

LIST_COLLECTIONS = (
    "SELECT DISTINCT collection FROM chunks WHERE app = $1 ORDER BY collection"
)

LIST_FILES = (
    "SELECT DISTINCT source FROM chunks WHERE app = $1 AND collection = $2 ORDER BY source"
)

COLLECTION_EXISTS = (
    "SELECT 1 FROM chunks WHERE app = $1 AND collection = $2 LIMIT 1"
)

DELETE_COLLECTION = "DELETE FROM chunks WHERE app = $1 AND collection = $2"

DELETE_FILE = "DELETE FROM chunks WHERE app = $1 AND collection = $2 AND source = $3"

# ── Ingestion ─────────────────────────────────────────────────────────────────

INSERT_CHUNK = """
    INSERT INTO chunks (id, app, collection, source, chunk_index, content, embedding, tsv)
    VALUES ($1, $2, $3, $4, $5, $6, $7, to_tsvector('simple', unaccent($6)))
"""

INSERT_QUESTION = """
    INSERT INTO chunk_questions (id, app, collection, source, chunk_id, chunk_index, question, embedding)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
"""

DELETE_QUESTIONS = "DELETE FROM chunk_questions WHERE app = $1 AND collection = $2 AND source = $3"

COUNT_QUESTIONS = "SELECT COUNT(*) FROM chunk_questions WHERE app = $1 AND collection = $2 AND source = $3"

# ── Retrieval ─────────────────────────────────────────────────────────────────

FULL_DOCUMENT = """
    SELECT content FROM chunks
    WHERE app = $1 AND collection = $2 AND source = $3
    ORDER BY chunk_index
"""

FILE_CHUNKS = """
    SELECT id, chunk_index, content FROM chunks
    WHERE app = $1 AND collection = $2 AND source = $3
    ORDER BY chunk_index
"""

COUNT_CHUNKS = "SELECT COUNT(*) FROM chunks WHERE app = $1 AND collection = $2 AND source = $3"

# Hybrid RRF: dense cosine + sparse FTS merged with Reciprocal Rank Fusion.
# $1=embedding  $2=app  $3=collection  $4=fetch_limit  $5=question
# $6=source_filter (NULL = no filter)  $7=final_limit
HYBRID_SEARCH = """
WITH semantic AS (
    SELECT id, source, content, chunk_index,
           ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS rank
    FROM chunks
    WHERE app = $2 AND collection = $3
      AND ($6::text IS NULL OR source = $6::text)
    ORDER BY embedding <=> $1::vector
    LIMIT $4
),
keyword AS (
    SELECT id, source, content, chunk_index,
           ROW_NUMBER() OVER (ORDER BY ts_rank(tsv, query) DESC) AS rank
    FROM chunks, websearch_to_tsquery('simple', unaccent($5)) AS query
    WHERE app = $2 AND collection = $3
      AND ($6::text IS NULL OR source = $6::text)
      AND tsv @@ query
    ORDER BY ts_rank(tsv, query) DESC
    LIMIT $4
),
combined AS (
    SELECT
        id,
        COALESCE(s.source,      k.source)      AS source,
        COALESCE(s.content,     k.content)     AS content,
        COALESCE(s.chunk_index, k.chunk_index) AS chunk_index,
        COALESCE(1.0 / (60.0 + s.rank), 0.0)
            + COALESCE(1.0 / (60.0 + k.rank), 0.0) AS rrf_score
    FROM semantic s
    FULL OUTER JOIN keyword k USING (id)
)
SELECT source, content, chunk_index, rrf_score
FROM combined
ORDER BY rrf_score DESC
LIMIT $7
"""

# Hypothetical-question search: dense ANN over generated questions, collapsed to
# the parent chunk they point to. Returns parent (source, chunk_index) identity —
# NOT the question text — so the result fuses with HYBRID_SEARCH on the same key
# and feeds the same window-expansion path. One chunk matched by several of its
# questions yields a single row (its closest question), so it never eats multiple
# result slots.
# $1=embedding  $2=app  $3=collection  $4=fetch_limit
# $5=source_filter (NULL = no filter)  $6=final_limit
QUESTION_SEARCH = """
WITH nearest AS (
    SELECT source, chunk_index, chunk_id,
           embedding <=> $1::vector AS dist
    FROM chunk_questions
    WHERE app = $2 AND collection = $3
      AND ($5::text IS NULL OR source = $5::text)
    ORDER BY embedding <=> $1::vector
    LIMIT $4
),
deduped AS (
    SELECT DISTINCT ON (chunk_id) source, chunk_index, dist
    FROM nearest
    ORDER BY chunk_id, dist
)
SELECT source, chunk_index
FROM deduped
ORDER BY dist
LIMIT $6
"""

# Fetch a contiguous slice of chunks from one document for window expansion.
# $1=app  $2=collection  $3=source  $4=index_low  $5=index_high
WINDOW_CHUNKS = """
    SELECT content
    FROM chunks
    WHERE app = $1 AND collection = $2 AND source = $3
      AND chunk_index BETWEEN $4 AND $5
    ORDER BY chunk_index
"""
