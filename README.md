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

- **Stateful Chat** — Persistent, session-based conversations stored in Redis with configurable TTL and automatic context-window trimming
- **RAG (Retrieval-Augmented Generation)** — Upload documents into named vector collections (single or batch) and ask grounded questions with source attribution; supports metadata filters and custom chunk sizes
- **File Processing** — Summarize or run custom tasks on uploaded PDFs, DOCX, and TXT files using a map-reduce pipeline with real-time streaming progress events
- **Multi-tenancy** — API key authentication with full data isolation between apps; each app only sees its own sessions and collections
- **Hashed API Keys** — Keys are stored as SHA-256 hashes in Redis; the plaintext is never persisted and is only returned once at generation time
- **Audit Log** — Redis-backed event log tracking key generation/revocation, auth failures, file operations, and admin actions — paginated per app or globally
- **Admin Panel** — HTTP Basic Auth-protected endpoints for provisioning/revoking API keys, wiping sessions, token usage stats, GPU monitoring, and audit log access
- **Rate Limiting** — Per-API-key, per-endpoint request limits to protect GPU resources (falls back to IP for unauthenticated routes)
- **Redis-backed GPU Concurrency** — Global token bucket in Redis (BLPOP/RPUSH) enforces `GPU_CONCURRENCY` across all workers and container replicas; requests block up to `GPU_WAIT_TIMEOUT` seconds (default 30 s) for a free slot, then return `503`
- **Usage Tracking** — Per-app prompt/completion token counters in Redis, exposed via admin endpoints
- **Async I/O** — Fully async stack: `redis.asyncio`, `AsyncOpenAI`, `asyncpg` for PostgreSQL/pgvector
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
  |  vectordb/       (pgvector)           |  <- Vector store + embeddings
  |  concurrency.py                     |  <- Redis GPU slot counter
  |  system/                            |  <- logger, limiter, .env loader
  └────────────────────────────────────┘
```

### Request Flow — Chat

1. Client sends `POST /general-requests/chat` with `X-API-Key` header
2. `verify_api_key` hashes the key with SHA-256 and looks it up in Redis → resolves to `app_name`
3. Session is retrieved from Redis (or created) using `chat:{app_name}:{session_id}`
4. User message is appended to history and sent to the LLM as a streaming request
5. Response is streamed back token-by-token; full response is saved to Redis on completion

### Request Flow — RAG Q&A

1. Client uploads a file via `POST /rag-db/upload` → text is extracted and stored in pgvector, scoped by `(app, collection)` columns in the `chunks` table
2. Client sends `POST /rag-db/ask` with a question, `collection_name`, and optional `n_results`
3. If a prior session exists, the question is **reformulated** into a standalone query using chat history
4. Top-N relevant chunks are retrieved from pgvector and injected as context
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

## Project Structure

```
PraixisEngine/
├── main.py                    # App entry point, FastAPI setup, lifespan, rate limit handler
├── Makefile                   # Docker shortcuts (up, up-local, down, down-local)
├── Dockerfile
├── docker-compose.yml         # App-only — bring your own Redis + Postgres
├── docker-compose.local.yml   # Overlay: adds bundled Postgres + Redis for local dev
├── tailwind.config.js         # Tailwind build config (brand colors, content paths)
├── pyproject.toml
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
    │   └── llm_runner.py      # Shared LLM execution, concurrent map-reduce, GPU slot management
    ├── models/
    │   └── schemas.py         # Pydantic request models
    ├── dependencies/
    │   └── security.py        # API key auth (SHA-256 lookup) + admin Basic Auth
    └── utils/
        ├── ai_client.py       # OpenAI-compatible client factory
        ├── concurrency.py     # Redis GPU slot counter, GPUBusyError
        ├── store/             # Redis client + data stores
        │   ├── client.py      # Shared async Redis client
        │   ├── sessions.py    # Chat session history
        │   ├── usage.py       # Per-app token usage counters
        │   ├── api_keys.py    # API key storage (SHA-256 hashed)
        │   └── audit.py       # Event log (Redis lists, newest-first pagination)
        ├── file_parser.py     # PDF / DOCX / TXT text extraction & chunking
        ├── system/            # Cross-cutting infrastructure
        │   ├── logger.py
        │   └── limiter.py     # SlowAPI rate limiter
        └── vectordb/          # pgvector connection, ingest, and retrieval
            ├── constants.py   # All SQL query strings
            ├── pool.py        # asyncpg connection pool lifecycle
            ├── embeddings.py  # fastembed text embedding
            ├── chunking.py    # Semantic and character chunking strategies
            ├── collections.py # Collection & file management
            ├── ingestion.py   # Chunk & index documents
            └── retrieval.py   # Hybrid semantic + FTS search
```

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
| `response_format` | `"text"` | `"text"` or `"json"` — instructs the LLM to return structured JSON |

Returns a streaming response. The first line is always `[SESSION_ID:<id>]` — save this to continue the conversation.

---

### File Summary — `POST /general-requests/file_summary`

Multipart form upload. Fields:

| Field | Default | Description |
|---|---|---|
| `file` | required | `.pdf`, `.docx`, or `.txt` — max **20 MB** |
| `task` | `"Summarize the key points of this document."` | Instruction for the AI |
| `tone` | `"Professional and objective"` | Desired response tone |

Returns `413 Request Entity Too Large` if the file exceeds 20 MB.

---

### Chat Session Management

| Method | Path | Description |
|---|---|---|
| `GET` | `/general-requests/chat/sessions/active` | List active session IDs for your app |
| `GET` | `/general-requests/chat/{session_id}` | Fetch the full message history for a session |
| `DELETE` | `/general-requests/chat/{session_id}` | Delete a session and its history |

---

### RAG Upload — `POST /rag-db/upload`

Accepts one or more files in a single request. Re-uploading a file that already exists in the collection replaces it automatically.

| Field | Default | Description |
|---|---|---|
| `files` | required | One or more `.pdf`, `.docx`, or `.txt` files — max **20 MB** each |
| `collection_name` | `"main"` | Target collection (alphanumeric/dash/underscore, 3–63 chars) |
| `chunking_strategy` | `"semantic"` | `"semantic"` — splits at natural topic boundaries using embeddings; `"character"` — fixed-size recursive splits |
| `chunk_size` | `2000` | Maximum characters per chunk (100–4000) |
| `chunk_overlap` | `150` | Overlap characters between chunks (0–500). Only applies when `chunking_strategy` is `"character"` |

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

Returns a **streaming response**. The first three lines are metadata headers, followed by the answer tokens:

```
[SESSION_ID:a1b2c3d4e5f6...]
[SEARCH_QUERY:the reformulated standalone query]
[SOURCES:filename1.pdf,filename2.pdf]
The answer begins streaming here...
```

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
| `DELETE` | `/rag-db/delete/{collection}` | Delete an entire collection |
| `DELETE` | `/rag-db/{collection}/files/{filename}` | Delete a single document from a collection |
| `GET` | `/rag-db/knowledge_base/{collection}/files/{filename}/summary` | 3-sentence summary of a document |
| `POST` | `/rag-db/knowledge_base/compare` | Bullet-point diff between two documents (JSON body: `collection_name`, `file_1`, `file_2`) |

---

### Admin Endpoints (Basic Auth)

All admin endpoints require HTTP Basic Auth (`ADMIN_USERNAME` / `ADMIN_PASSWORD`) except `/ping`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/system/ping` | Liveness check — no auth required |
| `GET` | `/api/system/auth/verify` | Validate admin credentials (used by the admin panel login); returns `{"ok": true}` |
| `GET` | `/api/system/health` | Aggregate health of Redis, vector store, and LLM backend |
| `GET` | `/api/system/health/redis` | Redis health only |
| `GET` | `/api/system/health/vectordb` | pgvector health only |
| `GET` | `/api/system/health/llm` | LLM backend health only |
| `GET` | `/api/system/stats` | Active sessions, collection count, total vector chunks |
| `GET` | `/api/system/keys` | List all provisioned keys (preview + created_at) and their app names |
| `POST` | `/api/system/keys/generate?app_name=` | Generate a new API key |
| `DELETE` | `/api/system/keys/revoke-by-hash?key_hash=` | Revoke a key by its stored SHA-256 hash |
| `DELETE` | `/api/system/sessions/{app_name}` | Force-wipe all active sessions for a specific app |
| `GET` | `/api/system/usage` | Token usage totals across all apps |
| `GET` | `/api/system/usage/{app_name}` | Token usage totals for a specific app |
| `GET` | `/api/system/gpu` | Current GPU slot usage (in-use / total / available) |
| `POST` | `/api/system/gpu/reset` | Rebuild the GPU slot queue to `GPU_CONCURRENCY` tokens (use after a crash leaked slots) |
| `GET` | `/api/system/audit?limit=100&offset=0` | Last N audit events across all apps, newest first |
| `GET` | `/api/system/audit/{app_name}` | Last N audit events for a specific app |
| `GET` | `/api/system/vector/search?app_name=&collection_name=&query=&n_results=5` | Semantic search inside a collection |
| `GET` | `/api/system/vector/collections` | List all vector collections across all apps |
| `GET` | `/api/system/vector/collections/{app_name}/{collection_name}/files` | List files in a collection |
| `DELETE` | `/api/system/vector/collections/{app_name}/{collection_name}` | Delete an entire collection |
| `DELETE` | `/api/system/vector/collections/{app_name}/{collection_name}/files` | Delete a specific file from a collection (query param: `filename`) |

---

## Rate Limits

All limits are per API key (falls back to IP for unauthenticated routes).

| Endpoint | Limit |
|---|---|
| `POST /general-requests/chat` | 10 / minute |
| `POST /general-requests/file_summary` | 5 / minute |
| `GET /general-requests/chat/sessions/active` | 60 / minute |
| `GET /general-requests/chat/{session_id}` | 60 / minute |
| `DELETE /general-requests/chat/{session_id}` | 30 / minute |
| `POST /rag-db/upload` | 15 / minute |
| `POST /rag-db/ask` | 30 / minute |
| `POST /rag-db/embed` | 60 / minute |
| `GET /rag-db/list` | 60 / minute |
| `GET /rag-db/{collection}/files` | 60 / minute |
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
| `GPU_CONCURRENCY` | `2` | Max simultaneous LLM calls (global — see below) |
| `GPU_WAIT_TIMEOUT` | `30` | Seconds a request waits for a free slot before returning 503 |
| `CHUNK_CONCURRENCY` | `4` | Max parallel chunk fan-out per `file_summary` map-reduce call (per-worker, internal) |

Slots are tokens in a Redis list (`gpu:slots`). Acquiring a slot is `BLPOP gpu:slots <timeout>`; releasing is `RPUSH gpu:slots 1`. Because Redis is the single source of truth, **`GPU_CONCURRENCY` is a true global limit** — running uvicorn with `--workers N` or scaling to multiple container replicas behind the same Redis still caps total in-flight LLM calls at `GPU_CONCURRENCY`. When all tokens are taken, `BLPOP` blocks the request for up to `GPU_WAIT_TIMEOUT` seconds; only after that timeout does the request fail with HTTP `503 Service Unavailable`. Callers may retry immediately or with a short backoff.

`CHUNK_CONCURRENCY` is enforced separately by an in-process `asyncio.Semaphore` inside the map-reduce pipeline and is per-worker — it limits how aggressively a single `file_summary` request fans out its chunks while it competes against other requests for the global GPU pool.

On startup the lifespan hook fills the queue **only if it has not already been sized for the current `GPU_CONCURRENCY`** (guarded by a sentinel key that stores the slot count), so a multi-worker or multi-replica deploy does not multiply the slot count, and changing `GPU_CONCURRENCY` in `.env` and restarting the container correctly resizes the queue. A hard process crash that releases tokens improperly will leak slots until `POST /api/system/gpu/reset` is called — that admin endpoint rebuilds the queue atomically and is visible to every worker on its next acquire.

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
