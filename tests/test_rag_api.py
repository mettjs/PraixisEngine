"""RAG endpoints (/rag-db/*) through the full app against embedded Chroma:
ingestion (multipart + raw text), inspection, search, ask (markers + SOURCES
escaping), the hypothetical-question index, summaries, compare, deletion."""
import time
import uuid

from conftest import DEFAULT_REPLY, EMBEDDING_DIMS

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
    assert body["score_type"] == "similarity"  # chroma backend: no hybrid FTS arm
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
    assert buffered.json() == {"filename": "doc.txt", "content": DEFAULT_REPLY}

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
