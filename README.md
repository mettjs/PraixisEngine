# PraixisEngine

A multi-tenant AI backend API that provides decoupled business logic for LLM-powered applications. Multiple client apps can connect using isolated API keys and get access to stateful chat, document Q&A (RAG), and file processing — all backed by a local or remote OpenAI-compatible LLM.

---

## Features

- **Stateful Chat** — Persistent, session-based conversations stored in Redis with configurable TTL and automatic context-window trimming
- **RAG (Retrieval-Augmented Generation)** — Upload documents into named vector collections (single or batch) and ask grounded questions with source attribution; supports metadata filters and custom chunk sizes
- **File Processing** — Summarize or run custom tasks on uploaded PDFs, DOCX, and TXT files using a map-reduce pipeline with real-time streaming progress events
- **Multi-tenancy** — API key authentication with full data isolation between apps; each app only sees its own sessions and collections
- **Admin Panel** — HTTP Basic Auth-protected endpoints for provisioning/revoking API keys, listing all keys, wiping app sessions, and per-app token usage stats
- **Rate Limiting** — Per-API-key, per-endpoint request limits to protect GPU resources (falls back to IP for unauthenticated routes)
- **GPU Concurrency Control** — Async semaphore limits simultaneous LLM calls; returns `503` immediately when at capacity rather than queuing
- **Usage Tracking** — Per-app prompt/completion token counters in Redis, exposed via admin endpoints
- **Async I/O** — Fully async stack: `redis.asyncio`, `AsyncOpenAI`, ChromaDB calls offloaded via `asyncio.to_thread`
- **Structured Output** — Optional `response_format: "json"` field on chat requests for machine-readable responses
- **Embeddings** — Direct embedding endpoint returns the raw vector for any text input

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
  |  chat_service.py            |  <- LLM calls, streaming, map-reduce
  |  rag_service.py             |  <- RAG pipeline, query reformulation
  └─────┬──────────────────────┘
        |
  ┌─────┴──────────────────────────────┐
  |           Utilities                 |
  |  ai_client.py  (OpenAI-compatible) |  <- LLM backend connection
  |  memory.py     (Redis)             |  <- Session storage & API keys
  |  vector_db.py  (ChromaDB)          |  <- Persistent vector store
  |  file_parser.py                    |  <- PDF / DOCX / TXT extraction
  |  concurrency.py                    |  <- GPU semaphore (BoundedSemaphore)
  └────────────────────────────────────┘
```

### Request Flow — Chat

1. Client sends `POST /general-requests/chat` with `X-API-Key` header
2. `verify_api_key` looks up the key in Redis → resolves to `app_name`
3. Session is retrieved from Redis (or created) using `chat:{app_name}:{session_id}`
4. User message is appended to history and sent to the LLM as a streaming request
5. Response is streamed back token-by-token; full response is saved to Redis on completion

### Request Flow — RAG Q&A

1. Client uploads a file via `POST /rag-db/upload` → text is extracted and chunked into ChromaDB under a scoped collection (`{app_name}_{collection_name}`)
2. Client sends `POST /rag-db/ask` with a question, `collection_name`, and optional `n_results`
3. If a prior session exists, the question is **reformulated** into a standalone query using chat history
4. Top-N relevant chunks are retrieved from ChromaDB and injected as context
5. Response is streamed back: metadata headers (`SESSION_ID`, `SEARCH_QUERY`, `SOURCES`) first, then answer tokens; full answer is saved to the session

### Large Document Pipeline (Map-Reduce)

For files that exceed a single context window (used by `/file_summary`):

```
Document
  └── Split into 1,500-word chunks
        └── MAP: Extract relevant info from each chunk
              └── REDUCE: Synthesize all extracted notes into the final result
```

---

## Project Structure

```
PraixisEngine/
├── main.py                    # App entry point, FastAPI setup, rate limit handler
├── pyproject.toml
├── src/
│   ├── routes/
│   │   ├── main_router.py     # Assembles all routers
│   │   ├── chat_router.py     # /general-requests endpoints
│   │   ├── rag_router.py      # /rag-db endpoints
│   │   └── admin_router.py    # /api/system endpoints
│   ├── controllers/
│   │   ├── chat_controller.py
│   │   ├── rag_controller.py
│   │   └── admin_controller.py
│   ├── services/
│   │   ├── chat_service.py    # LLM streaming, file summary map-reduce
│   │   └── rag_service.py     # RAG pipeline, query reformulation, comparison
│   ├── models/
│   │   └── schemas.py         # Pydantic request models
│   ├── dependencies/
│   │   └── security.py        # API key auth (Redis) + admin Basic Auth
│   └── utils/
│       ├── ai_client.py       # OpenAI-compatible client factory
│       ├── memory.py          # Redis session management
│       ├── vector_db.py       # ChromaDB CRUD operations
│       ├── file_parser.py     # PDF / DOCX / TXT text extraction & chunking
│       ├── concurrency.py     # GPU semaphore, GPUBusyError
│       ├── limiter.py         # SlowAPI rate limiter (IP-based)
│       ├── load_env.py        # .env loader
│       └── logger.py
└── logs/
```

---

## Setup

### Prerequisites

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) (package manager)
- A running Redis instance (local or [Upstash](https://upstash.com/))
- An OpenAI-compatible LLM server (e.g., [Ollama](https://ollama.com/), LM Studio, vLLM)

### Installation

```bash
git clone https://github.com/mettjs/PraixisEngine.git
cd PraixisEngine
uv sync
```

### Configuration

Copy `.env.example` to `.env` and fill in your values:

```env
# LLM Backend
AI_API_URL=http://localhost:8081      # Base URL of your OpenAI-compatible LLM server
AI_API_KEY=your-local-key            # API key for the LLM server (can be any string for local setups)
MODEL_NAME=gemma3:12b                # Model name as recognized by your LLM server

# GPU Concurrency
GPU_CONCURRENCY=2                    # Max simultaneous LLM calls (returns 503 when exceeded)

# Redis
REDIS_URL=redis://localhost:6379/0   # Use rediss:// for TLS (e.g., Upstash)

# Session
SESSION_TTL=86400                    # Session expiry in seconds (default: 24 hours)
MAX_HISTORY_PAIRS=20                 # Max user+assistant turns kept per session before oldest are trimmed

# ChromaDB
# CHROMA_PATH=./chroma_data          # Override the default ChromaDB storage path

# Admin
ADMIN_USERNAME=your_admin_username
ADMIN_PASSWORD=your_admin_password
```

### Running

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

API docs are available at `http://localhost:8080/docs`.

---

## Authentication

### Provisioning an API Key (Admin)

```bash
curl -X POST "http://localhost:8080/api/system/keys/generate?app_name=my-app" \
  -u admin_username:admin_password
```

Response:
```json
{
  "app_name": "my-app",
  "api_key": "praxis_...",
  "message": "Store this key safely! It will not be shown again."
}
```

### Using an API Key

Include the key in the `X-API-Key` header on all requests:

```bash
curl -X POST "http://localhost:8080/general-requests/chat" \
  -H "X-API-Key: praxis_..." \
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
  "session_id": "optional-existing-session-id"
}
```

Returns a streaming response. The first line is always `[SESSION_ID:<id>]` — save this to continue the conversation.

---

### File Summary — `POST /general-requests/file_summary`

Multipart form upload. Fields:

| Field | Default | Description |
|---|---|---|
| `file` | required | `.pdf`, `.docx`, or `.txt` — max **20 MB** |
| `task` | `Summarize the key points of this document.` | Instruction for the AI |
| `tone` | `Professional and objective` | Desired response tone |

Returns `413 Request Entity Too Large` if the file exceeds 20 MB.

---

### RAG Upload — `POST /rag-db/upload`

Accepts one or more files in a single request. Re-uploading a file that already exists in the collection replaces it automatically.

| Field | Default | Description |
|---|---|---|
| `files` | required | One or more `.pdf`, `.docx`, or `.txt` files — max **20 MB** each |
| `collection_name` | `main` | Target collection (alphanumeric/dash/underscore, 3–63 chars) |
| `chunk_size` | `1000` | Characters per chunk (100–4000) |
| `chunk_overlap` | `150` | Overlap characters between chunks (0–500) |

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
  "n_results": 5
}
```

| Field | Default | Description |
|---|---|---|
| `collection_name` | required | Target collection (alphanumeric/dash/underscore, 3–63 chars) |
| `question` | required | The question to ask |
| `session_id` | `null` | Existing session ID to continue a conversation |
| `n_results` | `5` | Number of context chunks to retrieve (1–20) |

Returns a **streaming response**. The first three lines are metadata headers, followed by the answer tokens:

```
[SESSION_ID:a1b2c3d4e5f6...]
[SEARCH_QUERY:the reformulated standalone query]
[SOURCES:filename1.pdf,filename2.pdf]
The answer begins streaming here...
```

Save the `SESSION_ID` to continue the conversation and benefit from automatic query reformulation on follow-up questions.

---

### Embed — `POST /rag-db/embed`

Returns the raw embedding vector for a text input using the same model the RAG pipeline uses internally (384 dimensions). Useful for client-side similarity search, semantic caching, routing logic, or storing vectors in an external DB that must stay consistent with ChromaDB. Does **not** call the LLM — pure CPU operation, hence the higher rate limit (60/min).

```json
{ "text": "What is the refund policy?" }
```

Response: `{"text": "...", "dimensions": 384, "embedding": [0.023, -0.147, ...]}`

---

### Other RAG Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/rag-db/list` | List all collections owned by your app |
| `GET` | `/rag-db/{collection}/files` | List files inside a collection |
| `DELETE` | `/rag-db/delete/{collection}` | Delete an entire collection |
| `DELETE` | `/rag-db/{collection}/files/{filename}` | Delete a single document from a collection |
| `GET` | `/rag-db/knowledge_base/{collection}/files/{filename}/summary` | 3-sentence summary of a document |
| `POST` | `/rag-db/knowledge_base/compare?collection_name=&file_1=&file_2=` | Bullet-point diff between two documents |

---

### Admin Endpoints (Basic Auth)

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/system/health` | Check Redis, ChromaDB, and LLM backend status |
| `GET` | `/api/system/stats` | Active sessions, collection count, total vector chunks |
| `GET` | `/api/system/keys` | List all provisioned API keys (masked) and their app names |
| `POST` | `/api/system/keys/generate?app_name=` | Generate a new API key |
| `DELETE` | `/api/system/keys/revoke?api_key=` | Permanently revoke an API key |
| `DELETE` | `/api/system/sessions/{app_name}` | Force-wipe all active sessions for a specific app |
| `GET` | `/api/system/usage` | Token usage totals across all apps |
| `GET` | `/api/system/usage/{app_name}` | Token usage totals for a specific app |

---

## Rate Limits

All limits are per IP address.

| Endpoint | Limit |
|---|---|
| `POST /general-requests/chat` | 10 / minute |
| `POST /general-requests/file_summary` | 5 / minute |
| `POST /rag-db/upload` | 15 / minute |
| `POST /rag-db/ask` | 30 / minute |
| `GET /rag-db/knowledge_base/.../summary` | 10 / minute |
| `POST /rag-db/knowledge_base/compare` | 5 / minute |
| `GET /rag-db/list` | 60 / minute |
| `GET`, `DELETE` collection/file endpoints | 60 / minute |

Exceeding a limit returns HTTP `429 Too Many Requests`.

### GPU Capacity

Endpoints that call the LLM (`/chat`, `/ask`, `/file_summary`, `/summarize`, `/compare`) share a semaphore pool sized by `GPU_CONCURRENCY` (default: `2`). When all slots are occupied, new requests immediately return HTTP `503 Service Unavailable` rather than queuing. Callers should retry with backoff.

---

## Multi-tenancy Model

All data is scoped to the `app_name` resolved from the API key:

- **Redis sessions** are stored as `chat:{app_name}:{session_id}`
- **ChromaDB collections** are stored as `{app_name}_{collection_name}` — the prefix is structural, so two apps using the same collection name get completely separate ChromaDB collections with no overlap. Access checks also verify the `app` metadata tag, returning `403 Access Denied` on any mismatch.
- **Admin operations** are separate and not scoped to any app

This means two different apps using the same collection name will have completely independent, isolated data stores.
