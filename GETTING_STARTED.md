# Getting Started

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) — package manager
- A running Redis instance (local, [Upstash](https://upstash.com/), or dedicated server)
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension (or use the bundled Docker Postgres service)
- An OpenAI-compatible LLM server ([Ollama](https://ollama.com/), LiteLLM, LM Studio, vLLM, etc.)

---

## Installation

```bash
git clone https://github.com/mettjs/PraixisEngine.git
cd PraixisEngine
uv sync
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```env
# LLM Backend — any OpenAI-compatible server
AI_API_URL=http://localhost:8081
AI_API_KEY=your-local-key
MODEL_NAME=gemma3:12b

# GPU Concurrency
GPU_CONCURRENCY=2       # max simultaneous interactive LLM calls
GPU_WAIT_TIMEOUT=30     # seconds to wait for a free slot before returning 503
CHUNK_CONCURRENCY=4     # parallel chunk fan-out per file_summary call

# Improved Search (hypothetical-question indexing) — opt-in per upload via improved_search=true
HQ_ENABLED=true        # global on/off switch for question generation
HQ_PER_CHUNK=5         # questions generated per chunk
HQ_GPU_CONCURRENCY=1   # GPU slots reserved for background generation (separate from GPU_CONCURRENCY;
                       #   total backend load = GPU_CONCURRENCY + HQ_GPU_CONCURRENCY. Set 0 to share the main pool)
HQ_GPU_WAIT_TIMEOUT=300 # seconds a background generation call waits for a reserved slot before skipping a chunk

# Redis
REDIS_URL=redis://localhost:6379/0

# Session
SESSION_TTL=86400       # seconds — default 24 hours
MAX_HISTORY_PAIRS=20    # user+assistant turns kept before oldest are trimmed

# PostgreSQL (pgvector)
POSTGRES_URL=postgresql://praixis:yourpassword@localhost:5432/praixis

# Admin panel credentials
ADMIN_USERNAME=your_admin_username
ADMIN_PASSWORD=your_admin_password

# Embedding model (optional — defaults work out of the box)
# Change only if using a different fastembed model; re-upload all collections after changing.
# EMBEDDING_DIMS must match the model's output dimension — startup fails fast if they disagree.
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DIMS=384
```

**Redis URL formats:**
- Local: `redis://localhost:6379/0`
- With password: `redis://:password@host:6379/0`
- TLS (Upstash, remote): `rediss://:password@host:6380/0`

---

## Running Locally

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

- Swagger UI: `http://localhost:8080/swagger/docs`
- ReDoc: `http://localhost:8080/docs`
- Admin panel: `http://localhost:8080/admin` (use `ADMIN_USERNAME` / `ADMIN_PASSWORD`)

---

## Running with Docker

Make sure Docker is running. The project includes a `Makefile` with two modes. `make` is built-in on macOS/Linux. On Windows, install it via [Chocolatey](https://chocolatey.org/) (`choco install make`) or use the manual commands shown below instead.

In both modes the API initializes the database schema on startup (creates the `vector` and `unaccent` extensions and the `chunks` and `chunk_questions` tables if they don't exist), so a fresh Postgres just works.

### Local stack — app + PostgreSQL + Redis in Docker

Use this when you want everything self-contained on one machine. Docker boots the app, a PostgreSQL/pgvector container, and a Redis container, wires them together, and persists data in named volumes.

```bash
make up-local
```

`REDIS_URL` and `POSTGRES_URL` are auto-overridden to point at the bundled containers, so you can leave them unset in `.env` for this mode. Local Postgres credentials are hardcoded to `praixis/praixis` and Postgres/Redis are exposed on the host at `5432`/`6379` for local debugging.

### App-only — bring your own Redis + Postgres

Use this when Redis and Postgres live elsewhere (managed services, separate servers, an existing cluster). Docker boots **only** the API container; you are responsible for providing reachable instances.

```bash
make up
```

Make sure your `.env` has the correct URLs before running. Postgres must have the `pgvector` extension available — the `pgvector/pgvector` image works out of the box:

```env
AI_API_URL=http://<llm-server-ip>:8081
REDIS_URL=redis://:password@<redis-server-ip>:6379/0
POSTGRES_URL=postgresql://<user>:<password>@<postgres-host>:5432/<db>
```

If you changed `EMBEDDING_MODEL`, pass the same value as a build arg so Docker pre-downloads the model during the image build instead of on the first request:

```bash
docker compose up --build --build-arg EMBEDDING_MODEL=your-model-name
```

### Tear down

```bash
make down        # matches make up
make down-local  # matches make up-local
```

### Manual commands (without make)

```bash
# Local stack (API + Postgres + Redis)
docker compose -f docker-compose.yml -f docker-compose.local.yml up --build

# App only (you provide REDIS_URL + POSTGRES_URL in .env)
docker compose up --build
```

---

## Provision Your First API Key

Once the app is running, create an API key for your client application:

```bash
curl -X POST "http://localhost:8080/api/system/keys/generate?app_name=my-app" \
  -u your_admin_username:your_admin_password
```

> `app_name` must match `^[a-zA-Z0-9_-]{3,63}$`.

Response:

```json
{
  "app_name": "my-app",
  "api_key": "praixis_...",
  "message": "Store this key safely. It will not be shown again."
}
```

**The key is only returned once.** It is stored as a SHA-256 hash in Redis — there is no way to retrieve the plaintext again. Save it immediately.

---

## Your First Request

Include the key in the `X-API-Key` header on every request:

```bash
curl -X POST "http://localhost:8080/general-requests/chat" \
  -H "X-API-Key: praixis_..." \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!", "session_id": null}'
```

The response streams token-by-token. The first line is always:

```
[SESSION_ID:a1b2c3d4e5f6...]
```

Save this ID and pass it as `session_id` in follow-up messages to continue the conversation.

---

## Next Steps

- Open the admin panel at `http://localhost:8080/admin` to manage keys, monitor GPU usage, and browse vector collections
- Upload documents and ask questions: see **RAG** endpoints in [README.md](README.md#api-reference)
- Check system health: `GET /api/system/health` (admin credentials required)
- Review all endpoints: [README.md → API Reference](README.md#api-reference)
