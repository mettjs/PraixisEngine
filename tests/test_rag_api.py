"""RAG endpoints (/rag-db/*) through the full app against the active vector
backend (chroma by default, pgvector when VECTOR_BACKEND says so): ingestion
(multipart + raw text), inspection, search, ask (markers + SOURCES escaping),
the hypothetical-question index, summaries, compare, deletion."""
import time
import uuid

import pytest

from conftest import DEFAULT_REPLY, EMBEDDING_DIMS, VECTOR_BACKEND

_LONG_TEXT = (
    "Employees accrue two vacation days per month of service. "
    "Unused days roll over for one calendar year and then expire. "
    "Leave requests must be approved by the direct supervisor. "
    "Sick leave requires a medical certificate from the third day. "
    "Parental leave follows the statutory national scheme. "
    "Remote work is allowed up to three days per week."
)


def _collection() -> str:
    return f"col-{uuid.uuid4().hex[:12]}"


def _upload_text(client, headers, collection, filename="handbook.txt",
                 text=_LONG_TEXT, **overrides):
    payload = {
        "text": text,
        "filename": filename,
        "collection_name": collection,
        "chunk_size": 100,
        "chunk_overlap": 20,
        "chunking_strategy": "character",
        **overrides,
    }
    return client.post("/rag-db/upload_text", json=payload, headers=headers)


def _wait_for_questions(client, headers, collection, filename, timeout=10.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        status = client.get(
            f"/rag-db/{collection}/files/{filename}/questions", headers=headers
        ).json()
        if not status["generation_pending"]:
            return status
        assert time.monotonic() < deadline, f"question generation never finished: {status}"
        time.sleep(0.05)


# ── Ingestion & inspection ────────────────────────────────────────────────────

def test_upload_text_then_list_and_inspect_chunks(client, headers):
    collection = _collection()
    body = _upload_text(client, headers, collection).json()
    assert body["status"] == "success"
    assert body["chunks_stored"] >= 2

    collections = client.get("/rag-db/list", headers=headers).json()
    assert collection in collections["active_collections"]

    files = client.get(f"/rag-db/{collection}/files", headers=headers).json()
    assert files["files_stored"] == ["handbook.txt"]

    chunks = client.get(f"/rag-db/{collection}/files/handbook.txt/chunks", headers=headers).json()
    assert chunks["total_chunks"] == body["chunks_stored"]
    indices = [chunk["chunk_index"] for chunk in chunks["chunks"]]
    assert indices == list(range(len(indices)))
    assert "vacation days" in chunks["chunks"][0]["content"]


def test_upload_text_validation(client, headers):
    collection = _collection()
    assert _upload_text(client, headers, collection, text="   ").status_code == 400
    assert _upload_text(client, headers, "ab").status_code == 422  # name too short
    response = _upload_text(client, headers, collection, chunk_size=100, chunk_overlap=100)
    assert response.status_code == 422  # overlap must be < size for character chunking


def test_multipart_upload_reports_per_file_results(client, headers):
    collection = _collection()
    response = client.post(
        "/rag-db/upload",
        files=[
            ("files", ("good.txt", _LONG_TEXT.encode(), "text/plain")),
            ("files", ("broken.pdf", b"%PDF-1.4 not a real pdf", "application/pdf")),
        ],
        data={"collection_name": collection, "chunking_strategy": "character"},
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["processed"] == 2
    assert body["succeeded"] == 1
    by_name = {entry["filename"]: entry for entry in body["results"]}
    assert by_name["good.txt"]["status"] == "success"
    assert by_name["broken.pdf"]["status"] == "error"
    assert "corrupted" in by_name["broken.pdf"]["detail"].lower()


# ── Search & embed ────────────────────────────────────────────────────────────

def test_search_returns_scored_chunks(client, headers):
    collection = _collection()
    _upload_text(client, headers, collection)
    response = client.post(
        "/rag-db/search",
        json={"collection_name": collection, "query": "how many vacation days", "n_results": 3},
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    # pgvector fuses a dense arm with an FTS arm and reports RRF; chroma is
    # pure vector search and reports raw similarity. The endpoint advertises
    # which, so the assertion follows the backend under test rather than
    # pinning one of them.
    expected_score_type = "rrf" if VECTOR_BACKEND == "pgvector" else "similarity"
    assert body["score_type"] == expected_score_type
    assert body["results"], "seeded collection must return hits"
    for hit in body["results"]:
        assert hit["source"] == "handbook.txt"
        assert hit["text"]
        assert 0.0 <= hit["score"] <= 1.0


def test_search_unknown_collection_is_404(client, headers):
    response = client.post(
        "/rag-db/search",
        json={"collection_name": "no-such-collection", "query": "anything"},
        headers=headers,
    )
    assert response.status_code == 404


def test_embed_returns_vector(client, headers):
    body = client.post("/rag-db/embed", json={"text": "hello"}, headers=headers).json()
    assert body["dimensions"] == EMBEDDING_DIMS
    assert len(body["embedding"]) == EMBEDDING_DIMS


# ── Ask ───────────────────────────────────────────────────────────────────────

def test_ask_buffered_returns_decoded_sources(client, headers, fake_llm):
    collection = _collection()
    # A filename with a comma exercises SOURCES escaping end to end.
    filename = "Q3, Final Report.pdf"
    _upload_text(client, headers, collection, filename=filename)

    response = client.post(
        "/rag-db/ask",
        json={"collection_name": collection, "question": "How many vacation days?", "stream": False},
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["content"] == DEFAULT_REPLY
    assert body["sources"] == [filename]
    # No session and no history → the question is used verbatim as the query.
    assert body["search_query"] == "How many vacation days?"
    assert body["session_id"]


def test_ask_streaming_emits_escaped_source_markers(client, headers, fake_llm):
    collection = _collection()
    _upload_text(client, headers, collection, filename="Q3, Final Report.pdf")

    response = client.post(
        "/rag-db/ask",
        json={"collection_name": collection, "question": "How many vacation days?", "stream": True},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "[SEARCH_QUERY:How many vacation days?]\n" in response.text
    assert "[SOURCES:Q3%2C Final Report.pdf]\n" in response.text
    assert response.text.endswith(DEFAULT_REPLY)


def test_ask_unknown_collection_is_404(client, headers):
    response = client.post(
        "/rag-db/ask",
        json={"collection_name": "no-such-collection", "question": "anything"},
        headers=headers,
    )
    assert response.status_code == 404


def test_ask_with_excluding_metadata_filter_is_404(client, headers):
    collection = _collection()
    _upload_text(client, headers, collection)
    response = client.post(
        "/rag-db/ask",
        json={
            "collection_name": collection,
            "question": "anything",
            "metadata_filter": {"source": "no-such-file.txt"},
        },
        headers=headers,
    )
    assert response.status_code == 404


# ── Hypothetical-question index ───────────────────────────────────────────────

def test_improved_search_generates_questions_in_background(client, headers, fake_llm):
    collection = _collection()
    fake_llm.queue("How many vacation days do I get?\nWho approves my leave?")
    body = _upload_text(
        client, headers, collection, text="Employees accrue two vacation days per month.",
        chunk_size=4000, improved_search=True,
    ).json()
    assert body["chunks_stored"] == 1

    status = _wait_for_questions(client, headers, collection, "handbook.txt")
    assert status["total_chunks"] == 1
    assert status["questions_stored"] == 2


def test_question_status_unknown_file_is_404(client, headers):
    collection = _collection()
    _upload_text(client, headers, collection)
    response = client.get(f"/rag-db/{collection}/files/ghost.txt/questions", headers=headers)
    assert response.status_code == 404


def test_regenerate_questions_conflicts_while_pending(client, headers, redis_state, fake_llm):
    collection = _collection()
    _upload_text(client, headers, collection, text="Short policy text.", chunk_size=4000)

    pending_key = f"hyq:pending:testapp:{collection}:handbook.txt"
    redis_state.set(pending_key, "1")
    response = client.post(f"/rag-db/{collection}/files/handbook.txt/questions", headers=headers)
    assert response.status_code == 409
    redis_state.delete(pending_key)

    fake_llm.queue("What does the policy say?")
    response = client.post(f"/rag-db/{collection}/files/handbook.txt/questions", headers=headers)
    assert response.status_code == 200
    assert response.json()["status"] == "scheduled"
    status = _wait_for_questions(client, headers, collection, "handbook.txt")
    assert status["questions_stored"] == 1


# ── Summary & compare ─────────────────────────────────────────────────────────

def test_document_summary_buffered_and_streaming(client, headers, fake_llm):
    collection = _collection()
    _upload_text(client, headers, collection, filename="doc.txt",
                 text="Short document body.", chunk_size=4000)

    buffered = client.get(
        f"/rag-db/knowledge_base/{collection}/files/doc.txt/summary", headers=headers
    )
    assert buffered.status_code == 200
    # The buffered body carries every marker the stream emits, so the model that
    # produced the summary is named here too.
    assert buffered.json() == {
        "filename": "doc.txt", "model": "default", "content": DEFAULT_REPLY,
    }

    streamed = client.get(
        f"/rag-db/knowledge_base/{collection}/files/doc.txt/summary",
        params={"stream": True},
        headers=headers,
    )
    assert streamed.status_code == 200
    assert streamed.text.startswith("[FILE:doc.txt]\n")


def test_compare_documents_buffered(client, headers, fake_llm):
    collection = _collection()
    _upload_text(client, headers, collection, filename="v1.txt",
                 text="Vacation: 20 days.", chunk_size=4000)
    _upload_text(client, headers, collection, filename="v2.txt",
                 text="Vacation: 25 days.", chunk_size=4000)

    response = client.post(
        "/rag-db/knowledge_base/compare",
        json={"collection_name": collection, "file_1": "v1.txt", "file_2": "v2.txt"},
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["file_1"] == "v1.txt"
    assert body["file_2"] == "v2.txt"
    assert body["content"] == DEFAULT_REPLY


# ── Deletion ──────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    VECTOR_BACKEND != "chroma",
    reason="chroma-specific rollback: pgvector inserts no rows on failure, so no "
           "collection comes into existence in the first place",
)
def test_failed_ingestion_leaves_no_empty_collection(client, headers, monkeypatch):
    """A failed insert must not strand the collection it just created.

    Chroma's get_or_create_collection creates the collection before any chunk
    is written, so an insert that raises would otherwise leave an empty
    collection behind — visible in listings, deletable with a 200 — breaking
    the same invariant the deletion path maintains.
    """
    import chromadb.api.models.Collection as _chroma_collection

    collection = _collection()

    def _boom(self, *args, **kwargs):
        raise RuntimeError("simulated chroma write failure")

    monkeypatch.setattr(_chroma_collection.Collection, "add", _boom)
    response = client.post(
        "/rag-db/upload_text",
        json={"text": "Vacation policy: two days per month.", "filename": "doomed.txt",
              "collection_name": collection, "chunking_strategy": "character"},
        headers=headers,
    )
    assert response.status_code >= 400, "the simulated write failure must surface"
    monkeypatch.undo()

    listed = client.get("/rag-db/list", headers=headers).json()["active_collections"]
    assert collection not in listed, "a failed upload must not leave an empty collection"
    assert client.delete(f"/rag-db/delete/{collection}", headers=headers).status_code == 404


def test_deleting_last_file_removes_the_collection(client, headers):
    """A collection exists exactly as long as it holds chunks.

    Identical on both backends: pgvector derives collections from `chunks`
    rows, and chroma drops the emptied collection to match. There is no
    create-collection endpoint, so an empty collection is not a state the API
    can represent.
    """
    collection = _collection()
    _upload_text(client, headers, collection, filename="only.txt")
    listed = client.get("/rag-db/list", headers=headers).json()["active_collections"]
    assert collection in listed

    assert client.delete(
        f"/rag-db/{collection}/files/only.txt", headers=headers
    ).status_code == 200

    listed = client.get("/rag-db/list", headers=headers).json()["active_collections"]
    assert collection not in listed, "emptying a collection must remove it"
    assert client.delete(f"/rag-db/delete/{collection}", headers=headers).status_code == 404


def test_delete_file_then_collection(client, headers):
    collection = _collection()
    _upload_text(client, headers, collection, filename="a.txt")
    _upload_text(client, headers, collection, filename="b.txt")

    assert client.delete(f"/rag-db/{collection}/files/a.txt", headers=headers).status_code == 200
    assert client.get(f"/rag-db/{collection}/files/a.txt/chunks", headers=headers).status_code == 404
    assert client.delete(f"/rag-db/{collection}/files/a.txt", headers=headers).status_code == 404

    assert client.delete(f"/rag-db/delete/{collection}", headers=headers).status_code == 200
    assert client.delete(f"/rag-db/delete/{collection}", headers=headers).status_code == 404
    collections = client.get("/rag-db/list", headers=headers).json()
    assert collection not in collections["active_collections"]


# ── Model selection ───────────────────────────────────────────────────────────

def test_ask_answers_on_the_requested_model_and_reformulates_on_the_utility_one(
    client, headers, fake_llm, multi_model
):
    """The two calls an /ask makes are routed independently: the answer goes to
    the model the caller picked, the query rewrite to the utility role. Billing
    a user's model for cheap chores is exactly what roles exist to prevent."""
    collection = _collection()
    _upload_text(client, headers, collection)
    # A session with history is what triggers reformulation at all.
    session_id = client.post(
        "/general-requests/chat", json={"prompt": "Tell me about leave", "stream": False},
        headers=headers,
    ).json()["session_id"]

    fake_llm.reset()
    response = client.post(
        "/rag-db/ask",
        json={
            "collection_name": collection,
            "question": "And how many are there?",
            "session_id": session_id,
            "model": "smart",
            "stream": False,
        },
        headers=headers,
    )
    assert response.status_code == 200
    reformulation, answer = fake_llm.calls[0], fake_llm.calls[-1]
    assert reformulation["model"] == "gemma4:e4b"   # roles.utility
    assert answer["model"] == "qwen3:32b"           # the request's model
    assert answer["stream"] is True


def test_ask_with_an_unknown_model_is_400(client, headers, fake_llm, multi_model):
    collection = _collection()
    _upload_text(client, headers, collection)
    response = client.post(
        "/rag-db/ask",
        json={"collection_name": collection, "question": "anything", "model": "gpt-9"},
        headers=headers,
    )
    assert response.status_code == 400
    # Rejected before the reformulation call, so an unknown id costs nothing.
    assert not fake_llm.calls


def test_compare_accepts_a_model(client, headers, fake_llm, multi_model):
    collection = _collection()
    _upload_text(client, headers, collection, filename="a.txt")
    _upload_text(client, headers, collection, filename="b.txt", text=_LONG_TEXT + " Overtime is paid.")
    response = client.post(
        "/rag-db/knowledge_base/compare",
        json={"collection_name": collection, "file_1": "a.txt", "file_2": "b.txt", "model": "fast"},
        headers=headers,
    )
    assert response.status_code == 200
    assert {call["model"] for call in fake_llm.calls} == {"gemma4:e4b"}


def test_document_summary_accepts_a_model_query_param(client, headers, fake_llm, multi_model):
    collection = _collection()
    _upload_text(client, headers, collection, filename="handbook.txt")
    response = client.get(
        f"/rag-db/knowledge_base/{collection}/files/handbook.txt/summary",
        params={"model": "fast"},
        headers=headers,
    )
    assert response.status_code == 200
    assert {call["model"] for call in fake_llm.calls} == {"gemma4:e4b"}


def test_ask_stream_emits_the_documented_marker_prefix(client, headers, fake_llm, multi_model):
    """The exact prefix the SDK decoders are written against.

    tests/marker_vectors.json pins how the three decoders *parse* this; this
    pins what the engine actually *writes*, which is the half the vectors
    cannot cover on their own.
    """
    collection = _collection()
    _upload_text(client, headers, collection, filename="handbook.txt")
    response = client.post(
        "/rag-db/ask",
        json={"collection_name": collection, "question": "How many vacation days?",
              "model": "smart", "stream": True},
        headers=headers,
    )
    session_id = response.text.split("]", 1)[0].removeprefix("[SESSION_ID:")
    assert response.text.startswith(
        f"[SESSION_ID:{session_id}]\n"
        f"[MODEL:smart]\n"
        f"[SEARCH_QUERY:How many vacation days?]\n"
        f"[SOURCES:handbook.txt]\n"
    ), response.text[:200]


def test_document_summary_stream_emits_file_then_model(client, headers, fake_llm, multi_model):
    collection = _collection()
    _upload_text(client, headers, collection, filename="handbook.txt")
    response = client.get(
        f"/rag-db/knowledge_base/{collection}/files/handbook.txt/summary",
        params={"model": "fast", "stream": "true"},
        headers=headers,
    )
    assert response.text.startswith("[FILE:handbook.txt]\n[MODEL:fast]\n"), response.text[:120]


def test_ask_stream_names_the_model_that_answered(client, headers, fake_llm, multi_model):
    collection = _collection()
    _upload_text(client, headers, collection)
    response = client.post(
        "/rag-db/ask",
        json={"collection_name": collection, "question": "How many vacation days?",
              "model": "smart", "stream": True},
        headers=headers,
    )
    assert "[MODEL:smart]\n" in response.text
    assert response.text.endswith(DEFAULT_REPLY)


def test_ask_buffered_reports_the_model_as_a_field(client, headers, fake_llm, multi_model):
    collection = _collection()
    _upload_text(client, headers, collection)
    body = client.post(
        "/rag-db/ask",
        json={"collection_name": collection, "question": "How many vacation days?",
              "model": "smart", "stream": False},
        headers=headers,
    ).json()
    assert body["model"] == "smart"


@pytest.mark.skipif(VECTOR_BACKEND != "chroma", reason="the rollback path is chroma-only")
def test_failed_upload_rollback_spares_a_concurrent_uploads_chunks(client, headers, monkeypatch):
    """Two uploads racing into a brand-new collection: the loser's rollback
    must not delete the winner's chunks.

    Ingestion runs in a worker thread, so the loser can read ``count() == 0``
    before the winner's insert lands. The winner is then told its file was
    stored, and a rollback trusting that stale reading would drop the whole
    collection out from under it. Reproduced by making the loser's collection
    report empty exactly once (its pre-insert check) and then fail its insert.
    """
    from src.utils.vectordb.chroma import client as chroma_client
    from src.utils.vectordb.chroma import ingestion

    collection = _collection()
    # The winner's upload succeeds normally and its chunks are in the store.
    assert _upload_text(client, headers, collection, filename="winner.txt").status_code == 200

    class _StaleEmptyCollection:
        """Delegates to the real collection, but reports empty on the first
        count() (the loser's pre-insert reading) and fails its insert."""

        def __init__(self, real):
            self._real = real
            self._counted = False

        def __getattr__(self, name):
            return getattr(self._real, name)

        def count(self):
            if not self._counted:
                self._counted = True
                return 0
            return self._real.count()

        def add(self, **_kwargs):
            raise RuntimeError("insert failed after a concurrent upload succeeded")

    real_get_client = ingestion.get_client

    class _WrappedClient:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def get_or_create_collection(self, **kwargs):
            return _StaleEmptyCollection(self._real.get_or_create_collection(**kwargs))

    monkeypatch.setattr(ingestion, "get_client", lambda: _WrappedClient(real_get_client()))

    dropped: list[str] = []
    real_drop = ingestion.drop_collection
    monkeypatch.setattr(ingestion, "drop_collection", lambda name, app: dropped.append(name))

    # The losing upload fails, as it must — the point is what it takes with it.
    failed = _upload_text(client, headers, collection, filename="loser.txt")
    assert failed.status_code in (200, 500)

    assert dropped == [], "rollback dropped a collection that a concurrent upload had filled"

    monkeypatch.undo()
    assert real_drop is ingestion.drop_collection
    # The winner's document is still there and still retrievable.
    files = client.get(f"/rag-db/{collection}/files", headers=headers).json()
    assert files["files_stored"] == ["winner.txt"], files
    assert chroma_client.get_owned_collection(collection, "testapp").count() > 0
