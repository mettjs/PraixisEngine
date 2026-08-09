"""Test bootstrap: fakes every external system before the app is imported.

The engine talks to three external services — Redis, an OpenAI-compatible LLM
backend, and an embedding model. The tests replace all three at import time,
BEFORE any ``src`` module binds its module-level clients:

* Redis → fakeredis. One shared ``FakeServer`` backs the engine's two async
  clients and the tests' own synchronous handle (the ``redis_state`` fixture),
  so tests can seed and inspect keys directly.
* The LLM → :class:`FakeLLM`, an in-process stand-in for ``AsyncOpenAI`` that
  supports streaming and non-streaming completions with usage blocks. Tests
  script it through the ``fake_llm`` fixture (``queue()`` replies or
  exceptions); unscripted calls return ``DEFAULT_REPLY``.
* Embeddings → a deterministic hash-based embedder, so no model download and
  no fastembed inference. Vectors are stable per text, which is all the
  chunking / retrieval code paths need.

The vector backend is real — not faked. It defaults to embedded Chroma on a
temp directory, so a bare ``uv run pytest`` needs no services. Set
``VECTOR_BACKEND=pgvector`` (plus ``POSTGRES_URL``) to run the identical suite
against a real Postgres + pgvector instead; CI does this as a matrix leg, since
pgvector is the production default. ``init_db`` creates its extensions and
schema idempotently, so the database needs no fixtures — only to exist.

Run with: ``uv run pytest``
       or: ``VECTOR_BACKEND=pgvector uv run pytest``
"""
import hashlib
import os
import re
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

# ── 1. Environment — must win over .env, so set before src.config is imported
# (python-dotenv does not override variables that are already set).

_CHROMA_TMP = tempfile.mkdtemp(prefix="praixis-test-chroma-")

# Which backend the suite exercises. Read from the ambient environment so CI can
# switch legs without editing this file; chroma stays the default so local runs
# stay service-free.
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "chroma").strip().lower()

# FakeEmbedder (section 4) emits one float per SHA-256 byte. pgvector's init_db
# validates the model's real output width against EMBEDDING_DIMS and refuses to
# build the schema on a mismatch, so the two must agree here — the app's default
# of 384 would abort the pgvector leg before a single test ran.
EMBEDDING_DIMS = 32

os.environ.update({
    "VECTOR_BACKEND": VECTOR_BACKEND,
    "CHROMA_PATH": _CHROMA_TMP,
    "POSTGRES_URL": os.getenv(
        "POSTGRES_URL", "postgresql://praixis:praixis@localhost:5432/praixis"
    ),
    "EMBEDDING_DIMS": str(EMBEDDING_DIMS),
    "REDIS_URL": "redis://fake-redis.invalid:6379/0",  # intercepted below
    "AI_API_URL": "http://fake-llm.invalid",
    "AI_API_KEY": "test-key",
    "MODEL_NAME": "fake-model",
    "ADMIN_USERNAME": "praixis-admin",
    "ADMIN_PASSWORD": "test-admin-secret",
    "SESSION_TTL": "3600",
    "CONTEXT_WINDOW": "8192",
    "GPU_CONCURRENCY": "2",
    "GPU_WAIT_TIMEOUT": "1",
    "HQ_GPU_CONCURRENCY": "1",
    "HQ_GPU_WAIT_TIMEOUT": "5",
    "CHUNK_CONCURRENCY": "4",
    "HQ_ENABLED": "true",
    "HQ_PER_CHUNK": "3",
})

ADMIN_AUTH = (os.environ["ADMIN_USERNAME"], os.environ["ADMIN_PASSWORD"])

# ── 2. Redis → fakeredis, before src.utils.store.client binds its clients.

import fakeredis  # noqa: E402
import fakeredis.aioredis  # noqa: E402
import redis.asyncio as _aioredis  # noqa: E402

_FAKE_SERVER = fakeredis.FakeServer()


def _fake_from_url(url, **kwargs):
    return fakeredis.aioredis.FakeRedis(
        server=_FAKE_SERVER, decode_responses=kwargs.get("decode_responses", False)
    )


_aioredis.Redis.from_url = _fake_from_url

# ── 3. LLM → FakeLLM, before llm_runner / session_stream / compaction bind
# their module-level ``_client = get_async_ai_client()``.

DEFAULT_REPLY = "This is a fake model reply."
_USAGE_PROMPT_TOKENS = 7
_USAGE_COMPLETION_TOKENS = 5


async def _stream_chunks(reply: str):
    usage = SimpleNamespace(
        prompt_tokens=_USAGE_PROMPT_TOKENS, completion_tokens=_USAGE_COMPLETION_TOKENS
    )
    for token in re.findall(r"\S+\s*", reply) or [reply]:
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=token))],
            usage=None,
        )
    # Final usage-only chunk, the include_usage shape OpenAI backends emit.
    yield SimpleNamespace(choices=[], usage=usage)


class FakeLLM:
    """Stands in for ``AsyncOpenAI``: chat.completions.create + models.list.

    ``queue(...)`` schedules the next replies in FIFO order; a queued
    ``Exception`` instance is raised instead of returned. With an empty queue
    every call answers ``DEFAULT_REPLY``. All calls are recorded in ``calls``.
    """

    def __init__(self) -> None:
        self.replies: list[object] = []
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )
        self.models = SimpleNamespace(list=self._models_list)

    def queue(self, *replies: object) -> None:
        self.replies.extend(replies)

    def reset(self) -> None:
        self.replies.clear()
        self.calls.clear()

    def with_options(self, **_kwargs) -> "FakeLLM":
        return self

    async def _models_list(self):
        return SimpleNamespace(data=[])

    async def _create(self, model, messages, stream=False, stream_options=None, **extra):
        self.calls.append({"messages": messages, "stream": stream, "extra": extra})
        reply = self.replies.pop(0) if self.replies else DEFAULT_REPLY
        if isinstance(reply, Exception):
            raise reply
        if stream:
            return _stream_chunks(reply)
        usage = SimpleNamespace(
            prompt_tokens=_USAGE_PROMPT_TOKENS, completion_tokens=_USAGE_COMPLETION_TOKENS
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
            usage=usage,
        )


FAKE_LLM = FakeLLM()

import src.utils.ai_client as _ai_client  # noqa: E402

_ai_client._client = FAKE_LLM  # type: ignore[assignment]

# ── 4. Embeddings → deterministic hash vectors (no model download).
# Width is EMBEDDING_DIMS, declared above because the env block needs it.


class FakeEmbedder:
    def embed(self, texts):
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            yield np.frombuffer(digest, dtype=np.uint8).astype(np.float32) + 1.0


import src.utils.vectordb.embeddings as _embeddings  # noqa: E402

_embeddings._embedder = FakeEmbedder()  # type: ignore[assignment]

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def client():
    """App under test, lifespan run (GPU pools filled, vector store initialized)."""
    from fastapi.testclient import TestClient
    from main import app

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def redis_state():
    """Synchronous handle on the same fake Redis the app uses."""
    return fakeredis.FakeStrictRedis(server=_FAKE_SERVER, decode_responses=True)


@pytest.fixture(scope="session")
def api_key(client) -> str:
    response = client.post(
        "/api/system/keys/generate", params={"app_name": "testapp"}, auth=ADMIN_AUTH
    )
    assert response.status_code == 200, response.text
    return response.json()["api_key"]


@pytest.fixture()
def headers(api_key) -> dict:
    return {"X-API-Key": api_key}


@pytest.fixture(scope="session")
def marker_vectors() -> dict:
    """The golden marker/escaping contract shared with the three SDKs.

    Authored here and vendored verbatim into praixis-python, praixis-node and
    praixis-go, where the same vectors are asserted against each SDK's own
    decoder. See the ``_comment`` block inside the file.
    """
    import json
    from pathlib import Path

    path = Path(__file__).parent / "marker_vectors.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture()
def fake_llm():
    FAKE_LLM.reset()
    yield FAKE_LLM
    FAKE_LLM.reset()


@pytest.fixture(autouse=True)
def _flush_throttles(redis_state):
    """Rate-limit and auth-fail windows must not leak across tests."""
    yield
    for pattern in ("ratelimit:*", "authfail:*"):
        keys = redis_state.keys(pattern)
        if keys:
            redis_state.delete(*keys)
