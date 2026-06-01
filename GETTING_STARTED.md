# Getting Started

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) — package manager
- A running Redis instance (local, [Upstash](https://upstash.com/), or dedicated server)
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
GPU_CONCURRENCY=2       # max simultaneous LLM calls
GPU_WAIT_TIMEOUT=30     # seconds to wait for a free slot before returning 503
CHUNK_CONCURRENCY=4     # parallel chunk fan-out per file_summary call

# Redis
REDIS_URL=redis://localhost:6379/0

# Session
SESSION_TTL=86400       # seconds — default 24 hours
MAX_HISTORY_PAIRS=20    # user+assistant turns kept before oldest are trimmed

# ChromaDB — optional, defaults to ./chroma_data
# CHROMA_PATH=./chroma_data

# Admin panel credentials
ADMIN_USERNAME=your_admin_username
ADMIN_PASSWORD=your_admin_password
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

Vector data is stored by embedded ChromaDB inside the API container at `/app/chroma_data` and is persisted across container restarts by the `chroma_data` named volume defined in `docker-compose.yml`. No external vector-DB service is required in either mode.

### Local stack — app + Redis in Docker

Use this when you want everything self-contained on one machine. Docker boots both the app and a Redis container, wires them together, and persists both Chroma and Redis data in named volumes.

```bash
make up-local
```

`REDIS_URL` is auto-overridden to point at the bundled Redis container, so you can leave it unset in `.env` for this mode. Redis is exposed on the host at `6379` for local debugging.

### App-only — bring your own Redis

Use this when Redis (and/or the LLM) lives elsewhere (managed service, separate server, existing cluster). Docker boots **only** the API container; you are responsible for providing a reachable Redis instance.

```bash
make up
```

Make sure your `.env` has the correct URLs before running:

```env
AI_API_URL=http://<llm-server-ip>:8081
REDIS_URL=redis://:password@<redis-server-ip>:6379/0
```

### Tear down

```bash
make down        # matches make up
make down-local  # matches make up-local
```

### Manual commands (without make)

```bash
# Local stack (API + Redis)
docker compose -f docker-compose.yml -f docker-compose.local.yml up --build

# App only (you provide REDIS_URL in .env)
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
