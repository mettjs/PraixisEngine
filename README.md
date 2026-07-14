# PraixisEngine

A multi-tenant AI backend API that provides decoupled business logic for LLM-powered applications. Multiple client apps can connect using isolated API keys and get access to stateful chat, document Q&A (RAG), and file processing — all backed by a local or remote OpenAI-compatible LLM.

**→ New here? Start with [GETTING_STARTED.md](GETTING_STARTED.md)**

---

## Client SDKs

Official SDKs are available for connecting to PraixisEngine from your application:

**Go**
```bash
go get github.com/mettjs/praixis-go
```

**Node.js**
```bash
npm install praixis
```

**Python**
```bash
# Synchronous client
pip install praixis

# Async client
pip install "praixis[async]"
```

---

## Features

- **Stateful Chat** — Persistent, session-based conversations stored in Redis with configurable TTL and automatic context management: when a session approaches the `CONTEXT_WINDOW` token budget, older exchanges are compacted into an LLM-written summary (recent turns kept verbatim) instead of being dropped; compaction is also available on demand per session
- **RAG (Retrieval-Augmented Generation)** — Upload documents into named vector collections (single or batch) and ask grounded questions with source attribution; supports metadata filters and custom chunk sizes. On the pgvector backend retrieval is hybrid: dense vector similarity fused with full-text keyword search via Reciprocal Rank Fusion
- **Pluggable Vector Store** — One `VECTOR_BACKEND` setting selects the retrieval backend: **pgvector** (PostgreSQL: hybrid dense + full-text search, the default) or **chroma** (embedded ChromaDB: zero extra infrastructure, pure vector search). Both share the same embeddings, chunking, API surface, and Improved Search; see [Vector Store Backends](#vector-store-backends)
- **Improved Search (Hypothetical-Question Indexing)** — Opt-in per upload (`improved_search=true`). After a document is stored, an LLM generates the natural-language questions each chunk answers; those questions are embedded and indexed so plain, conversational queries match better against formal or technical source text (closing the "genre gap" between how people ask and how documents are written). Generation runs in the background on a **dedicated GPU pool** so it never competes with live chat/RAG traffic; the document is searchable immediately and question matching improves once generation finishes
- **File Processing** — Summarize or run custom tasks on uploaded PDFs, DOCX, and TXT files using a map-reduce pipeline with real-time streaming progress events
- **Multi-tenancy** — API key authentication with full data isolation between apps; each app only sees its own sessions and collections
- **Hashed API Keys** — Keys are stored as SHA-256 hashes in Redis; the plaintext is never persisted and is only returned once at generation time
- **Audit Log** — Redis-backed event log tracking key generation/revocation, auth failures, file operations, and admin actions — paginated per app or globally
- **Admin Panel** — HTTP Basic Auth-protected endpoints for provisioning/revoking API keys, wiping sessions, token usage stats, GPU monitoring, and audit log access
- **Rate Limiting** — Per-API-key, per-endpoint request limits to protect GPU resources (falls back to IP for unauthenticated routes)
- **Redis-backed GPU Concurrency** — Global token buckets in Redis (BLPOP/RPUSH) enforce concurrency across all workers and container replicas. Two independent pools: a shared `GPU_CONCURRENCY` pool for interactive calls (requests block up to `GPU_WAIT_TIMEOUT` seconds, default 30 s, then return `503`) and a separate `HQ_GPU_CONCURRENCY` pool reserved for background question generation so it never starves live traffic
- **Usage Tracking** — Prompt/completion token counters in Redis, per app (exposed via admin endpoints) and per session (exposed to clients via `GET /general-requests/chat/{session_id}/usage`); session counters expire with the session
- **Async I/O** — Fully async stack: `redis.asyncio`, `AsyncOpenAI`, `asyncpg` for PostgreSQL/pgvector (Chroma's sync client is isolated in worker threads)
- **Structured Output** — Optional `response_format: "json"` field on chat requests for machine-readable responses
- **Embeddings** — Direct embedding endpoint returns the raw vector for any text input using the same multilingual model (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensions) the RAG pipeline uses internally; model is configurable via `EMBEDDING_MODEL` (must be paired with `EMBEDDING_DIMS` set to the model's output dimension — startup validates the pair and fails fast if they disagree)

---

## Architecture

```
Client App (with X-API-Key)
        |
        v
  FastAPI (main.py)
        |
  ┌─────┴──────────────────────┐
  |           Routes            |
  |  /general-requests          |  <- Chat & file processing
  |  /rag-db                    |  <- Vector DB / Q&A
  |  /api/system                |  <- Admin (Basic Auth)
  └─────┬──────────────────────┘
        |
  ┌─────┴──────────────────────┐
  |         Services            |
  |  chat_service.py            |  <- Chat streaming, file summary
  |  rag_service.py             |  <- RAG pipeline, query reformulation
  |  llm_runner.py              |  <- Shared LLM execution, map-reduce, GPU slots
  └─────┬──────────────────────┘
        |
  ┌─────┴──────────────────────────────┐
  |           Utilities                 |
  |  ai_client.py    (OpenAI-compatible)|  <- LLM backend connection
  |  store/          (Redis)            |  <- Client, sessions, usage, keys, audit
  |  vectordb/  (pgvector | chroma)     |  <- Vector store + embeddings
  |  concurrency.py                     |  <- Redis GPU slot counter
  |  system/                            |  <- logger, limiter, .env loader
  └────────────────────────────────────┘
```

### Request Flow — Chat

1. Client sends `POST /general-requests/chat` with `X-API-Key` header
2. `verify_api_key` hashes the key with SHA-256 and looks it up in Redis → resolves to `app_name`
3. Session is retrieved from Redis (or created) using `chat:{app_name}:{session_id}`
4. User message is appended to history; if the history is near the `CONTEXT_WINDOW` budget, older exchanges are auto-compacted into a summary first
5. History is sent to the LLM as a streaming request
6. Response is streamed back token-by-token; full response is saved to Redis on completion and token usage is recorded per app and per session

### Request Flow — RAG Q&A

1. Client uploads a file via `POST /rag-db/upload` → text is extracted and stored in the active vector backend, scoped by app and collection. If `improved_search=true`, hypothetical questions are generated in the background and indexed alongside (pgvector: `chunk_questions` table; Chroma: a parallel questions collection), each linked back to its parent chunk
2. Client sends `POST /rag-db/ask` with a question, `collection_name`, and optional `n_results`
3. If a prior session exists, the question is **reformulated** into a standalone query using chat history
4. Candidates are retrieved over the source text (hybrid dense + keyword on pgvector; dense on Chroma) and — when a question index exists — by dense search over the generated questions (de-duplicated to the parent chunk); the ranked lists are fused with Reciprocal Rank Fusion, then the top-N are window-expanded and injected as context
5. Response is streamed back: metadata headers (`SESSION_ID`, `SEARCH_QUERY`, `SOURCES`) first, then answer tokens; full answer is saved to the session

### Large Document Pipeline (Map-Reduce)

For files that exceed a single context window (used by `/file_summary`):

```
Document
  └── Split into chunks (~9,000 chars, respecting paragraph/sentence boundaries)
        └── MAP: Extract relevant info from each chunk
              └── REDUCE: Synthesize all extracted notes into the final result
```

---

## Vector Store Backends

The retrieval backend is selected once per deployment with `VECTOR_BACKEND`; everything else (API surface, embeddings, chunking, Improved Search, admin panel) is identical.

| | `pgvector` (default) | `chroma` |
|---|---|---|
| Storage | PostgreSQL + [pgvector](https://github.com/pgvector/pgvector) | Embedded [ChromaDB](https://www.trychroma.com/), persisted to `CHROMA_PATH` |
| Extra infrastructure | A Postgres server | None |
| Document search | **Hybrid**: dense vectors + full-text keyword search fused with RRF | Dense vectors only |
| Improved Search (question index) | Yes (`chunk_questions` table, FK-cascaded) | Yes (parallel questions collection) |
| Best for | Production: better retrieval on keyword-ish queries (names, article numbers, codes) | Getting started, small single-node deployments |
| Docker mode | `make up-local` (bundled Postgres + Redis) | `make up-chroma` (bundled Redis only) |

Pick the backend before uploading documents — the two stores do not share data, and there is no migration tool between them. Switching `VECTOR_BACKEND` on an existing deployment means re-uploading your collections.

```env
VECTOR_BACKEND=pgvector   # or: chroma
POSTGRES_URL=postgresql://praixis:yourpassword@localhost:5432/praixis  # pgvector only
CHROMA_PATH=./chroma_data                                              # chroma only
```

---

## LLM Backends

The app talks to the LLM through a single OpenAI-compatible endpoint (`AI_API_URL` + `MODEL_NAME`), so any compliant server works. Point it at one you already run, or bring up a bundled backend with a Docker overlay that composes on top of either vector stack.

| | External (default) | vLLM overlay | LiteLLM + Ollama overlay |
|---|---|---|---|
| File | — (just set `AI_API_URL`) | `docker-compose.vllm.yml` | `docker-compose.litellm.yml` (+ `litellm_config.yaml`) |
| Adds | nothing — you run the server | a `vllm` service serving one model | a `litellm` proxy in front of an `ollama` engine |
| GPU | whatever the server needs | **NVIDIA, Ampere or newer** (won't run on a Mac) | none required — Ollama runs on CPU or a host GPU |
| Strength | reuse existing infrastructure | high concurrency via continuous batching | simplest local/Mac path; quantized models |
| Model var | `MODEL_NAME` | `VLLM_MODEL` (HF repo id) | `LITELLM_MODEL` (Ollama tag) |
| Make target | — | `make up-local-vllm` / `up-chroma-vllm` | `make up-local-litellm` / `up-chroma-litellm` |

Both overlays override `AI_API_URL`/`MODEL_NAME` to point at the in-network service, so you only set the model. Compose them with whichever vector overlay you use, e.g. `make up-chroma-vllm`.

**Which engine?** vLLM and Ollama are both inference *engines* (they run the model); LiteLLM is a *proxy/router* that runs no models itself. vLLM's continuous batching gives multi-x throughput under concurrent load, but it's built around FP16/BF16 and tensor cores — it expects Ampere-or-newer hardware and won't meaningfully help (and may not even run) on older Pascal-class cards like the Tesla P40. Ollama (llama.cpp) runs quantized GGUF models well on CPUs and older GPUs and is the easy local option. Keep LiteLLM in front when you serve several models/servers and want one endpoint with routing and fallbacks — it's orthogonal to vLLM, not replaced by it.

---

## Project Structure

```
PraixisEngine/
├── main.py                    # App entry point, FastAPI setup, lifespan, rate limit handler
├── Makefile                   # Docker shortcuts (up, up-local, up-chroma + matching down-*)
├── Dockerfile
├── docker-compose.yml         # App-only — bring your own Redis + Postgres
├── docker-compose.local.yml   # Overlay: adds bundled Postgres + Redis for local dev
├── docker-compose.chroma.yml  # Overlay: bundled Redis + embedded Chroma — no Postgres
├── tailwind.config.js         # Tailwind build config (brand colors, content paths)
├── pyproject.toml
├── tests/                     # Pytest suite — fakeredis + fake LLM/embedder + temp Chroma (see Testing)
│   ├── conftest.py            # Test bootstrap: fakes Redis, the LLM, and embeddings before app import
│   ├── test_units.py          # Marker parsing, SOURCES escaping, file parsing, compaction sizing
│   ├── test_chat_api.py       # /general-requests endpoints, auth, rate limits, GPU exhaustion
│   ├── test_rag_api.py        # /rag-db endpoints against embedded Chroma, question index
│   └── test_admin_api.py      # /api/system endpoints, key lifecycle, usage, audit
└── src/
    ├── config.py             # Single source of truth: loads .env and parses all env vars
    ├── admin_panel/           # Browser-based admin UI (served at /admin)
    │   ├── base.html          # Root template — assembles all includes
    │   ├── components/        # Shared UI fragments (sidebar, header, login, toast, icons)
    │   ├── views/             # Page panels (dashboard, keys, usage, vector, audit)
    │   ├── modals/            # Dialog overlays (generate key, revoke, wipe sessions, etc.)
    │   └── static/
    │       ├── css/           # admin.css, layout.css, buttons.css, forms.css, modal.css
    │       ├── js/            # admin.js (core), dashboard.js, keys.js, usage.js,
    │       │                  #   audit.js, vector.js, helpers.js
    │       └── img/           # logo.png
    ├── routes/
    │   ├── main_router.py     # Assembles all routers
    │   ├── chat_router.py     # /general-requests endpoints
    │   ├── rag_router.py      # /rag-db endpoints
    │   ├── admin_router.py    # /api/system endpoints
    │   └── ui_router.py       # Serves /admin and /static/*
    ├── controllers/
    │   ├── chat_controller.py
    │   ├── rag_controller.py
    │   └── admin_controller.py
    ├── services/
    │   ├── chat_service.py    # LLM streaming, file summary map-reduce
    │   ├── rag_service.py     # RAG pipeline, query reformulation, comparison
    │   ├── compaction.py      # Conversation compaction (auto + on-demand summary of older turns)
    │   └── llm_runner.py      # Shared LLM execution, concurrent map-reduce, GPU slot management
    ├── models/
    │   └── schemas.py         # Pydantic request models
    ├── dependencies/
    │   └── security.py        # API key auth (SHA-256 lookup) + admin Basic Auth
    └── utils/
        ├── ai_client.py       # OpenAI-compatible client factory
        ├── concurrency.py     # Redis GPU slot counters (interactive + reserved question-gen pools), GPUBusyError
        ├── store/             # Redis client + data stores
        │   ├── client.py      # Shared async Redis client
        │   ├── sessions.py    # Chat session history
        │   ├── usage.py       # Per-app and per-session token usage counters
        │   ├── api_keys.py    # API key storage (SHA-256 hashed)
        │   └── audit.py       # Event log (Redis lists, newest-first pagination)
        ├── file_parser.py     # PDF / DOCX / TXT text extraction & chunking
        ├── system/            # Cross-cutting infrastructure
        │   ├── logger.py
        │   └── limiter.py     # Async Redis fixed-window rate limiter
        └── vectordb/          # Vector store: shared contract + per-backend packages
            ├── __init__.py    # Factory: resolves VECTOR_BACKEND to a store singleton
            ├── base.py        # VectorStore ABC + capability flags (supports_hybrid, supports_questions)
            ├── embeddings.py  # fastembed text embedding (shared by both backends)
            ├── chunking.py    # Semantic and character chunking strategies (shared)
            ├── fusion.py      # RRF rank fusion + context-window merging (shared)
            ├── questions.py   # Hypothetical-question generation (Improved Search), reserved GPU pool
            ├── pg/            # PostgreSQL + pgvector backend
            │   ├── constants.py   # All SQL (chunks + chunk_questions schema & queries)
            │   ├── pool.py        # asyncpg connection pool lifecycle
            │   ├── collections.py # Collection & file management
            │   ├── ingestion.py   # Chunk & index documents
            │   ├── retrieval.py   # Hybrid (dense + FTS) search fused with question index via RRF
            │   └── store.py       # PgVectorStore — VectorStore implementation
            └── chroma/        # Embedded ChromaDB backend
                ├── client.py      # Persistent client lifecycle, tenant scoping, ownership checks
                ├── collections.py # Collection & file management
                ├── ingestion.py   # Chunk & index documents, question storage
                ├── retrieval.py   # Dense search fused with question index via RRF
                └── store.py       # ChromaStore — VectorStore implementation
```

---

## Testing

```bash
uv run pytest
```

The suite needs no external services: Redis is replaced by fakeredis, the LLM
backend and the embedding model by deterministic in-process fakes (all wired in
`tests/conftest.py` before the app is imported), and the vector store is real
embedded Chroma on a temp directory — the same code paths a
`VECTOR_BACKEND=chroma` deployment runs. The pgvector backend is exercised only
by its shared contract; backend-specific SQL needs a real Postgres and is out
of scope here.

---

## Authentication

### API Keys

API keys are stored as SHA-256 hashes in Redis. The plaintext key is only returned once at generation time and is never retrievable again.

When a request arrives, the incoming key is hashed and looked up in Redis. A failed lookup logs an `AUTH_FAIL` audit event (with a key preview, never the full key).

**Migration note:** If you are upgrading from a version that stored keys in plaintext, all existing keys will stop working and must be regenerated.

Provisioning a key:

```bash
curl -X POST "http://localhost:8080/api/system/keys/generate?app_name=my-app" \
  -u admin_username:admin_password
```

Response:

```json
{
  "app_name": "my-app",
  "api_key": "praixis_...",
  "message": "Store this key safely. It will not be shown again."
}
```

### Using a Key

Include it in the `X-API-Key` header on every request:

```bash
curl -X POST "http://localhost:8080/general-requests/chat" \
  -H "X-API-Key: praixis_..." \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!", "session_id": null}'
```

---

## API Reference

### Chat — `POST /general-requests/chat`

```json
{
  "prompt": "What is the refund policy?",
  "system_prompt": "You are a helpful support agent.",
  "session_id": "optional-existing-session-id",
  "response_format": "text"
}
```

| Field | Default | Description |
|---|---|---|
| `prompt` | required | The user message |
| `system_prompt` | `"You are a helpful institutional assistant."` | Only applied when creating a new session; ignored on existing sessions |
| `session_id` | `null` | Existing session ID to continue a conversation |
| `stream` | `true` | `true` — stream tokens as `text/event-stream`; `false` — return one buffered JSON body |
| `response_format` | `"text"` | `"text"` or `"json"` — instructs the LLM to return structured JSON |

By default returns a streaming response. The first line is always `[SESSION_ID:<id>]` — save this to continue the conversation. With `stream=false` the same reply arrives as one JSON body:

```json
{"session_id": "a1b2c3d4e5f6...", "content": "The full reply..."}
```

---

### File Summary — `POST /general-requests/file_summary`

Multipart form upload. Fields:

| Field | Default | Description |
|---|---|---|
| `file` | required | PDF, DOCX, or TXT — max **20 MB** |
| `task` | `"Summarize the key points of this document."` | Instruction for the AI |
| `tone` | `"Professional and objective"` | Desired response tone |
| `stream` | `true` | `true` — stream tokens as `text/event-stream`; `false` — return one buffered JSON body |
| `response_format` | `"text"` | `"text"` or `"json"` — instructs the LLM to return structured JSON (applied to the final synthesis only) |

When streaming, the first line is `[FILE:<filename>]`, then for multi-chunk documents `[PROGRESS:mapping N chunks]`, a `[PROGRESS:mapped k/N chunks]` tick as each chunk completes, and `[PROGRESS:reducing N chunks]`, followed by the result tokens. With `stream=false` the same result arrives as a single JSON body (progress markers are dropped):

```json
{"filename": "report.pdf", "content": "The document outlines..."}
```

Returns `413 Request Entity Too Large` if the file exceeds 20 MB. The format is detected from the filename extension, falling back to the part's `Content-Type` header, then to magic bytes (see [RAG Upload](#rag-upload--post-rag-dbupload)).

---

### Chat Session Management

| Method | Path | Description |
|---|---|---|
| `GET` | `/general-requests/chat/sessions/active` | List active session IDs for your app |
| `GET` | `/general-requests/chat/{session_id}` | Fetch the full message history for a session |
| `GET` | `/general-requests/chat/{session_id}/usage` | Token usage for the session: `requests`, `prompt_tokens`, `completion_tokens`, `total_tokens`, plus `estimated_context_tokens` (current history size vs `CONTEXT_WINDOW`) |
| `POST` | `/general-requests/chat/{session_id}/compact` | Compact the conversation now: older exchanges are folded into an LLM-written summary, the last 3 exchanges stay verbatim. Returns message/token counts before and after. `400` if there is nothing to fold yet |
| `DELETE` | `/general-requests/chat/{session_id}/last` | Undo the last exchange: removes the most recent user message and the assistant reply that followed it (or just the user message if generation failed). Returns the `undone_prompt` so clients can retry or regenerate. Compaction summaries are kept. `400` if there is no user message left |
| `DELETE` | `/general-requests/chat/{session_id}` | Delete a session, its history, and its usage counters |

Compaction also happens **automatically**: when a session's history reaches ~80% of the `CONTEXT_WINDOW` token budget (estimated at ~4 chars/token), it is compacted transparently before the next reply, so long conversations retain their context instead of dropping their oldest turns.

---

### RAG Upload — `POST /rag-db/upload`

Accepts one or more files in a single request. Re-uploading a file that already exists in the collection replaces it automatically.

The format of each file is resolved as: filename extension → the part's declared `Content-Type` header (`application/octet-stream` is treated as "no information") → magic bytes (`%PDF-`, `PK\x03\x04`). The filename itself is always required — it is the document's stored identity, used by the delete/summarize/compare endpoints.

| Field | Default | Description |
|---|---|---|
| `files` | required | One or more PDF, DOCX, or TXT files — max **20 MB** each |
| `collection_name` | `"main"` | Target collection (alphanumeric/dash/underscore, 3–63 chars) |
| `chunking_strategy` | `"semantic"` | `"semantic"` — splits at natural topic boundaries using embeddings; `"character"` — fixed-size recursive splits |
| `chunk_size` | `2000` | Maximum characters per chunk (100–4000) |
| `chunk_overlap` | `150` | Overlap characters between chunks (0–500). Only applies when `chunking_strategy` is `"character"` |
| `improved_search` | `false` | Enable hypothetical-question indexing for this document (see [Improved Search](#improved-search-hypothetical-question-indexing) below) |

> **Improved Search caveats.** When `improved_search=true`, question generation runs **in the background after the upload response returns** — the document is searchable immediately, but natural-language matching improves only once generation completes. While it runs, one or more GPU slots from the **reserved question-generation pool** (`HQ_GPU_CONCURRENCY`) are occupied, raising backend load. Results vary by document, but conversational-query accuracy usually improves. Requires the global `HQ_ENABLED` flag to be on. If a document is deleted or re-uploaded mid-generation, the in-flight pass is discarded and the newer upload regenerates.

Returns per-file results:

```json
{
  "collection_name": "company-policies",
  "processed": 2,
  "succeeded": 2,
  "results": [
    {"filename": "policy_a.pdf", "status": "success"},
    {"filename": "policy_b.pdf", "status": "success"}
  ]
}
```

---

### RAG Upload (raw text) — `POST /rag-db/upload_text`

Ingest text directly, without wrapping it in a file — for callers that already hold the content (scraped pages, database records, generated documents). Runs the exact same pipeline as `/upload` after text extraction, so the stored document is indistinguishable from a file upload and works with every other endpoint (`ask`, `chunks`, `summary`, `compare`, delete, question indexing).

```json
{
  "text": "Full document text here… (max 20 MB)",
  "filename": "faq-2026.txt",
  "collection_name": "main",
  "chunking_strategy": "semantic",
  "chunk_size": 2000,
  "chunk_overlap": 150,
  "improved_search": false
}
```

`filename` is required — it becomes the document's stored identity, exactly as with `/upload`, and re-using one replaces the prior document. Returns `{"status", "collection_name", "filename", "chunks_stored", "improved_search"}`.

---

### RAG Ask — `POST /rag-db/ask`

```json
{
  "collection_name": "company-policies",
  "question": "What is the vacation accrual rate?",
  "session_id": "optional-existing-session-id",
  "n_results": 5,
  "metadata_filter": null
}
```

| Field | Default | Description |
|---|---|---|
| `collection_name` | required | Target collection |
| `question` | required | The question to ask |
| `session_id` | `null` | Existing session ID for follow-up questions with automatic query reformulation |
| `n_results` | `5` | Number of context chunks to retrieve (1–20) |
| `system_prompt` | `null` | Optional system prompt override; falls back to the built-in RAG instruction when omitted |
| `metadata_filter` | `null` | Optional metadata filter dict (e.g. `{"source": "file.pdf"}`) |
| `stream` | `true` | `true` — stream tokens as `text/event-stream`; `false` — return one buffered JSON body |
| `response_format` | `"text"` | `"text"` or `"json"` — instructs the LLM to return structured JSON |

By default returns a **streaming response**. The first three lines are metadata headers, followed by the answer tokens:

```
[SESSION_ID:a1b2c3d4e5f6...]
[SEARCH_QUERY:the reformulated standalone query]
[SOURCES:filename1.pdf,filename2.pdf]
The answer begins streaming here...
```

With `stream=false` the same metadata and answer arrive as one JSON body:

```json
{
  "session_id": "a1b2c3d4e5f6...",
  "search_query": "the reformulated standalone query",
  "sources": ["filename1.pdf", "filename2.pdf"],
  "content": "The full answer..."
}
```

---

### RAG Search — `POST /rag-db/search`

Retrieval **only** — returns the ranked raw chunks that match a query, with **no LLM call and no synthesis**. Use this when the caller wants to do its own reasoning over the evidence (e.g. an orchestrator fusing these chunks with other sources) instead of receiving a finished answer. Unlike `/rag-db/ask`, this endpoint does **not** reformulate the query and does **not** hold a GPU slot. It runs the same backend-agnostic ranked search the admin panel uses: hybrid (dense + full-text, fused with RRF) on pgvector, dense on Chroma.

```json
{
  "collection_name": "company-policies",
  "query": "vacation accrual rate",
  "n_results": 5
}
```

| Field | Default | Description |
|---|---|---|
| `collection_name` | required | Target collection (alphanumeric/dash/underscore, 3–63 chars) |
| `query` | required | The search text |
| `n_results` | `5` | Number of chunks to return (1–20) |

Returns one buffered JSON body (no streaming):

```json
{
  "collection_name": "company-policies",
  "query": "vacation accrual rate",
  "n_results": 5,
  "results": [
    {"source": "hr-handbook.docx", "text": "Employees accrue 1.25 days per month...", "score": 0.0312}
  ],
  "score_type": "rrf"
}
```

`score_type` tells you how to read `score`: `"rrf"` on the hybrid (pgvector) backend returns small Reciprocal Rank Fusion values where higher is better; `"similarity"` on the dense-only (Chroma) backend returns a 0–1 similarity. Returns `404` if the collection does not exist.

---

### Improved Search (Hypothetical-Question Indexing)

Formal documents (laws, policies, technical manuals) are written in a different register than the way people ask about them. A citizen asks *"can they fire me for being pregnant?"* while the statute says *"termination of the employment contract on grounds of pregnancy is prohibited."* Their embeddings sit far apart even though they're about the same thing — so pure semantic search can miss the right passage.

**Improved Search** closes that gap from the document side. When you upload with `improved_search=true`:

1. The document is chunked, embedded, and stored as usual — **searchable immediately**.
2. In the **background**, an LLM reads each chunk and writes the plain-language questions a non-expert would ask that the chunk answers.
3. Those questions are embedded and indexed separately (pgvector: `chunk_questions` table; Chroma: a parallel questions collection), each pointing back to its parent chunk.

At query time, the user's question is matched **question-to-question** (same register, much tighter similarity) in addition to the normal document search; the two result sets are fused and de-duplicated to the parent chunk. One chunk surfaced by several of its questions still counts once. This works identically on both vector backends.

**Operational notes:**

- **Opt-in and additive.** It's off by default and set per upload. It's an additional ranking signal, not a replacement — on queries that already match well it changes nothing; its value shows on conversational, low-keyword-overlap questions over large corpora.
- **Dedicated GPU pool.** Generation draws from a separate slot pool (`HQ_GPU_CONCURRENCY`, default `1`) so it never starves interactive chat/RAG, and is never starved by it. Total concurrent LLM calls a deployment issues is `GPU_CONCURRENCY + HQ_GPU_CONCURRENCY` — size your backend accordingly. See [GPU Concurrency](#gpu-concurrency).
- **Best-effort.** Generation has no progress tracking. A process restart mid-generation leaves a document partially enriched — re-upload it to finish. A per-chunk generation failure is logged and skipped, not fatal.
- **Global kill-switch.** Set `HQ_ENABLED=false` to disable generation everywhere regardless of the per-upload flag. Retrieval transparently ignores the question index when a document has none.

---

### Embed — `POST /rag-db/embed`

Returns the raw embedding vector for a text input using the same model the RAG pipeline uses internally (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensions by default). Does **not** call the LLM.

```json
{ "text": "What is the refund policy?" }
```

Response:

```json
{"text": "...", "dimensions": 384, "embedding": [0.023, -0.147, ...]}
```

---

### Other RAG Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/rag-db/list` | List all collections owned by your app |
| `GET` | `/rag-db/{collection}/files` | List files inside a collection |
| `GET` | `/rag-db/{collection}/files/{filename}/chunks` | Inspect a document's stored chunks (`chunk_index` + `content`, in order) — see exactly what retrieval sees when debugging |
| `GET` | `/rag-db/{collection}/files/{filename}/questions` | Question-index status for a document: `total_chunks`, `questions_stored`, and `generation_pending` (true while a background pass is running) |
| `POST` | `/rag-db/{collection}/files/{filename}/questions` | Backfill or rebuild the hypothetical-question index for an already-stored document — `improved_search` is no longer locked in at upload time. Regenerates in the background and swaps out the old questions only once the new pass has results, so a failed pass never strips a working index (poll the `GET` for progress). `409` while a pass is already running, `400` when `HQ_ENABLED=false` |
| `DELETE` | `/rag-db/delete/{collection}` | Delete an entire collection |
| `DELETE` | `/rag-db/{collection}/files/{filename}` | Delete a single document from a collection |
| `GET` | `/rag-db/knowledge_base/{collection}/files/{filename}/summary` | 3-sentence summary of a document. Returns `{"filename", "content"}` by default; pass `?stream=true` for a token stream (`[FILE:...]` header, then `[PROGRESS:...]` lines, then tokens). Also accepts `?response_format=json` |
| `POST` | `/rag-db/knowledge_base/compare` | Bullet-point diff between two documents (JSON body: `collection_name`, `file_1`, `file_2`, optional `stream` and `response_format`). Returns `{"file_1", "file_2", "content"}` by default; with `"stream": true`, streams `[PROGRESS:...]` lines then tokens |

---

### Admin Endpoints (Basic Auth)

All admin endpoints require HTTP Basic Auth (`ADMIN_USERNAME` / `ADMIN_PASSWORD`) except `/ping`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/system/ping` | Liveness check — no auth required |
| `GET` | `/api/system/auth/verify` | Validate admin credentials (used by the admin panel login); returns `{"ok": true}` |
| `GET` | `/api/system/health` | Aggregate health of Redis, vector store, and LLM backend |
| `GET` | `/api/system/health/redis` | Redis health only |
| `GET` | `/api/system/health/vectordb` | Vector store health only (active backend) |
| `GET` | `/api/system/health/llm` | LLM backend health only |
| `GET` | `/api/system/stats` | Active sessions, collection count, total vector chunks |
| `GET` | `/api/system/keys` | List all provisioned keys (preview + created_at) and their app names |
| `POST` | `/api/system/keys/generate?app_name=` | Generate a new API key |
| `POST` | `/api/system/keys/rotate?key_hash=` | Rotate a key: issues a new key for the owning app, then revokes the old one — no zero-key window |
| `DELETE` | `/api/system/keys/revoke-by-hash?key_hash=` | Revoke a key by its stored SHA-256 hash |
| `DELETE` | `/api/system/sessions/{app_name}` | Force-wipe all active sessions for a specific app |
| `GET` | `/api/system/usage` | Token usage totals across all apps |
| `GET` | `/api/system/usage/{app_name}` | Token usage totals for a specific app |
| `GET` | `/api/system/usage/{app_name}/daily?days=7` | Per-UTC-day usage for the last N days (1–90), newest first; daily buckets are kept 90 days |
| `GET` | `/api/system/gpu` | Current GPU slot usage for both pools — interactive (`slots_*`) and reserved question-generation (`hq_slots_*`) |
| `POST` | `/api/system/gpu/reset` | Rebuild both GPU slot queues to `GPU_CONCURRENCY` / `HQ_GPU_CONCURRENCY` tokens (use after a crash leaked slots) |
| `GET` | `/api/system/audit?limit=100&offset=0` | Last N audit events across all apps, newest first |
| `GET` | `/api/system/audit/{app_name}` | Last N audit events for a specific app |
| `GET` | `/api/system/vector/search?app_name=&collection_name=&query=&n_results=5` | Semantic search inside a collection |
| `GET` | `/api/system/vector/collections` | List all vector collections across all apps |
| `GET` | `/api/system/vector/collections/{app_name}/{collection_name}/files` | List files in a collection |
| `DELETE` | `/api/system/vector/collections/{app_name}/{collection_name}` | Delete an entire collection |
| `DELETE` | `/api/system/vector/collections/{app_name}/{collection_name}/files` | Delete a specific file from a collection (query param: `filename`) |

---

## Rate Limits

All limits are per API key (falls back to client IP for values without the issued `praixis_` prefix). Requests with an **invalid** key are additionally capped at 30/minute per IP before returning `403` — beyond that they get `429`.

| Endpoint | Limit |
|---|---|
| `POST /general-requests/chat` | 10 / minute |
| `POST /general-requests/file_summary` | 5 / minute |
| `GET /general-requests/chat/sessions/active` | 60 / minute |
| `GET /general-requests/chat/{session_id}` | 60 / minute |
| `GET /general-requests/chat/{session_id}/usage` | 60 / minute |
| `POST /general-requests/chat/{session_id}/compact` | 10 / minute |
| `DELETE /general-requests/chat/{session_id}/last` | 30 / minute |
| `DELETE /general-requests/chat/{session_id}` | 30 / minute |
| `POST /rag-db/upload` | 15 / minute |
| `POST /rag-db/upload_text` | 15 / minute |
| `POST /rag-db/ask` | 30 / minute |
| `POST /rag-db/search` | 30 / minute |
| `POST /rag-db/embed` | 60 / minute |
| `GET /rag-db/list` | 60 / minute |
| `GET /rag-db/{collection}/files` | 60 / minute |
| `GET /rag-db/{collection}/files/{filename}/chunks` | 30 / minute |
| `GET /rag-db/{collection}/files/{filename}/questions` | 60 / minute |
| `POST /rag-db/{collection}/files/{filename}/questions` | 10 / minute |
| `GET /rag-db/knowledge_base/.../summary` | 10 / minute |
| `POST /rag-db/knowledge_base/compare` | 5 / minute |
| `DELETE /rag-db/delete/{collection}` | 20 / minute |
| `DELETE /rag-db/{collection}/files/{filename}` | 20 / minute |

Exceeding a limit returns HTTP `429 Too Many Requests`.

---

## GPU Concurrency

Endpoints that call the LLM (`/chat`, `/ask`, `/file_summary`, `/summarize`, `/compare`) share a Redis-backed token bucket sized by `GPU_CONCURRENCY` (default: `2`).

| Env var | Default | Description |
|---|---|---|
| `GPU_CONCURRENCY` | `2` | Max simultaneous **interactive** LLM calls (global — see below) |
| `GPU_WAIT_TIMEOUT` | `30` | Seconds an interactive request waits for a free slot before returning 503 |
| `CHUNK_CONCURRENCY` | `4` | Max parallel chunk fan-out per `file_summary` map-reduce call (per-worker, internal) |
| `HQ_GPU_CONCURRENCY` | `1` | Slots reserved exclusively for background hypothetical-question generation (separate pool) |
| `HQ_GPU_WAIT_TIMEOUT` | `300` | Seconds a background generation call waits for a reserved slot before skipping a chunk |

There are **two independent token-bucket pools**, each a Redis list:

- `gpu:slots` — the shared pool for interactive, user-facing calls (`/chat`, `/ask`, `/file_summary`, `/summarize`, `/compare`), sized by `GPU_CONCURRENCY`.
- `gpu:hq_slots` — reserved exclusively for background question generation (Improved Search), sized by `HQ_GPU_CONCURRENCY`.

Acquiring a slot is `BLPOP <queue> <timeout>`; releasing is `RPUSH <queue> 1`. Because Redis is the single source of truth, **each cap is a true global limit** — running uvicorn with `--workers N` or scaling to multiple container replicas behind the same Redis still caps total in-flight calls per pool. When all tokens are taken, `BLPOP` blocks for up to the pool's timeout; only after that does an interactive request fail with HTTP `503 Service Unavailable` (callers may retry with a short backoff), while a background generation call simply skips that chunk.

**Why a separate pool?** Generating questions for a large document is many LLM calls. Routing them through the shared pool would let one upload starve live chat/RAG into 503s (or get starved itself and silently drop questions). The reserved pool guarantees interactive traffic always keeps its `GPU_CONCURRENCY` slots and generation always has capacity. The trade-off: the **total** concurrent load your LLM backend may see is `GPU_CONCURRENCY + HQ_GPU_CONCURRENCY`. Tune for your hardware:
- *Backend has headroom* → keep them additive (e.g. `2` + `1` = up to `3` concurrent).
- *Fixed total budget* → split it (e.g. `GPU_CONCURRENCY=1` + `HQ_GPU_CONCURRENCY=1` keeps the total at `2`, carving one slot out for questions).
- *Disable the reserve* → `HQ_GPU_CONCURRENCY=0` makes generation fall back to the shared pool (it will then contend with interactive traffic).

`CHUNK_CONCURRENCY` is enforced separately by an in-process `asyncio.Semaphore` inside the map-reduce pipeline and is per-worker — it limits how aggressively a single `file_summary` request fans out its chunks while it competes for the shared GPU pool.

On startup the lifespan hook fills each queue **only if it has not already been sized for its current count** (guarded by a per-pool sentinel key), so a multi-worker or multi-replica deploy does not multiply the counts, and changing a count in `.env` and restarting the container correctly resizes that queue. A hard process crash that releases tokens improperly will leak slots until `POST /api/system/gpu/reset` is called — that admin endpoint rebuilds **both** queues atomically and is visible to every worker on its next acquire. `GET /api/system/gpu` reports usage for both pools (`slots_*` and `hq_slots_*`).

---

## Audit Log

Key security and data-mutation events are written to Redis lists and served newest-first via the admin audit endpoints. Both a global list and per-app lists are maintained, capped at 10,000 entries each.

Recorded events:

| Event | Trigger |
|---|---|
| `AUTH_FAIL` | Invalid or missing API key on any request |
| `KEY_GENERATED` | Admin created a new API key |
| `KEY_REVOKED` | Admin revoked a key |
| `SESSION_WIPED` | Admin force-deleted an app's sessions |
| `GPU_RESET` | Admin manually reset the GPU counter |
| `FILE_UPLOADED` | Document added to a RAG collection |
| `FILE_DELETED` | Document removed from a RAG collection |
| `COLLECTION_DELETED` | Entire RAG collection deleted |

Chat content and RAG query text are deliberately not logged.

---

## Admin Panel UI

A browser-based control panel is served at `GET /admin`. It provides the same functionality as the admin API endpoints through a visual interface:

- **Overview** — live service health, active session count, vector chunk count, GPU slot utilization
- **API Keys** — generate keys, revoke keys, wipe app sessions
- **Token Usage** — per-app prompt/completion token breakdown
- **Vector DB** — browse collections, delete collections or files, run semantic search queries
- **Audit Log** — paginated event log with per-app filtering

Open it in a browser and authenticate with `ADMIN_USERNAME` / `ADMIN_PASSWORD`:

```
http://localhost:8080/admin
```

Alpine.js (3.14.3) and Tailwind CSS are vendored locally — the admin panel makes no external requests at runtime. Admin credentials are held in `sessionStorage` and are cleared when the tab is closed.

---

## Multi-tenancy Model

All data is scoped to the `app_name` resolved from the API key:

- **Redis sessions** are stored as `chat:{app_name}:{session_id}`
- **Vector collections** are rows in the `chunks` table scoped by `(app, collection)` composite columns — two apps using the same collection name get completely separate data with no overlap. Every query filters by `app`, so cross-tenant access returns `404` (collection not found) and never leaks existence of another app's data
- **Usage counters** are stored as `usage:{app_name}:*`
- **Audit logs** are stored under `audit:{app_name}` in addition to the global `audit:global` list
- **Admin operations** are separate and not scoped to any app
