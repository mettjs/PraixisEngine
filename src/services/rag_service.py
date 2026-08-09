import asyncio
from collections.abc import AsyncGenerator
from src.models.registry import ModelSpec, resolve_role
from src.utils.file_parser import chunk_text
from src.services.llm_runner import call_llm, map_calls_iter, stream_llm
from src.services.session_stream import open_user_turn, stream_assistant_turn
from src.utils.system.streaming import encode_source_list


async def generate_rag_answer(
    question: str,
    app_name: str,
    context_chunks: list[dict[str, str]],
    search_query: str,
    spec: ModelSpec,
    system_prompt: str | None = None,
    session_id: str | None = None,
    response_format: str = "text",
    model_was_explicit: bool = False,
) -> AsyncGenerator[str, None]:
    """Generates an answer to a question based only on the context provided by the collection.

    The GPU slot is acquired by the controller and released by the
    ``SlotReleasingStreamingResponse`` wrapping this generator.
    """
    if not system_prompt:
        system_prompt = (
            "You are an expert institutional analyst. "
            "Answer in the same language as the user. "
            "Use ONLY the provided context to answer the user's question. "
            "If the answer is not contained in the context, explain that the answer cannot be found in the document, do not fabricate any information."
        )

    active_session_id, history = await open_user_turn(
        app_name=app_name,
        user_message=question,
        spec=spec,
        session_id=session_id,
        system_prompt=system_prompt,
        model_was_explicit=model_was_explicit,
    )

    formatted_chunks = [f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk in context_chunks]
    context_text = "\n\n---\n\n".join(formatted_chunks)
    augmented_question = f"Context:\n{context_text}\n\nQuestion: {question}"

    # The LLM sees the augmented question; the session stores the original one.
    temp_history = history[:-1] + [{"role": "user", "content": augmented_question}]

    yield f"[SESSION_ID:{active_session_id}]\n"
    yield f"[MODEL:{spec.id}]\n"
    yield f"[SEARCH_QUERY:{search_query}]\n"
    unique_sources = list({chunk["source"] for chunk in context_chunks})
    yield f"[SOURCES:{encode_source_list(unique_sources)}]\n"

    extra: dict = {}
    if response_format == "json":
        extra["response_format"] = {"type": "json_object"}

    async for token in stream_assistant_turn(
        messages=temp_history,
        app_name=app_name,
        session_id=active_session_id,
        history=history,
        spec=spec,
        extra=extra,
    ):
        yield token


async def reformulate_query(
    history: list, latest_question: str, app_name: str, session_id: str | None = None
) -> str:
    """Uses recent chat history to rewrite follow-up questions into standalone queries.

    Returns the question unchanged when it introduces a new, independent topic so that
    stale context from prior exchanges does not pollute an unrelated search query.

    Runs on the ``utility`` role, not on the model answering the question:
    rewriting a query is cheap work and should stay on the cheap model even
    when the answer itself is routed elsewhere.
    """
    if len(history) <= 1:
        return latest_question

    reformulation_prompt = (
        "You are a search query optimizer. Given a conversation history and the user's latest question, "
        "decide whether the question is a follow-up that depends on the prior conversation "
        "(e.g. it uses pronouns like 'it', 'them', 'this', 'that', or implicitly references something already discussed).\n\n"
        "- If it IS a follow-up: rewrite it as a fully standalone search query by resolving those references.\n"
        "- If it is a NEW, independent topic with no contextual dependency on the history: return the question EXACTLY as written.\n\n"
        "Do NOT answer the question. Return ONLY the (possibly rewritten) query, nothing else."
    )

    recent = history[1:][-6:]  # last 3 exchanges, skip system prompt
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent])
    user_msg = f"History:\n{history_text}\n\nLatest Question: {latest_question}"

    content = await call_llm(
        [
            {"role": "system", "content": reformulation_prompt},
            {"role": "user", "content": user_msg},
        ],
        app_name,
        resolve_role("utility"),
        session_id=session_id,
    )
    return content.strip() if content else latest_question


async def _map_reduce_stream(
    text: str,
    map_prompt: str,
    reduce_prompt: str,
    single_chunk_prompt: str,
    app_name: str,
    spec: ModelSpec,
    extra: dict | None = None,
) -> AsyncGenerator[str, None]:
    """Streaming map-reduce: yields ``[PROGRESS:...]`` markers through the map
    phase, then streams the final synthesis token-by-token. ``extra`` is applied
    only to that final call — the map phase produces intermediate notes that are
    concatenated, so it stays text. Each LLM call self-acquires its GPU slot, so
    the caller must NOT pre-acquire."""
    chunks = chunk_text(text=text, max_words_per_chunk=1500)
    total = len(chunks)

    if total == 1:
        messages = [{"role": "user", "content": f"{single_chunk_prompt}\n\n{chunks[0]}"}]
        async for token in stream_llm(messages, app_name, spec, extra=extra):
            yield token
        return

    yield f"[PROGRESS:mapping {total} chunks]\n"
    extracted: list[str] = []
    async for item in map_calls_iter(
        [[{"role": "user", "content": f"{map_prompt}\n\n{chunk}"}] for chunk in chunks],
        app_name,
        spec,
    ):
        if isinstance(item, list):
            extracted = item
        else:
            yield f"[PROGRESS:mapped {item}/{total} chunks]\n"
    yield f"[PROGRESS:reducing {total} chunks]\n"
    reduce_messages = [{"role": "user", "content": f"{reduce_prompt}\n\n" + "\n\n".join(extracted)}]
    async for token in stream_llm(reduce_messages, app_name, spec, extra=extra):
        yield token


async def generate_summary(
    document_text: str, app_name: str, spec: ModelSpec, response_format: str = "text"
) -> AsyncGenerator[str, None]:
    """Summarizes a document using map-reduce, streaming the final summary."""
    extra: dict = {}
    if response_format == "json":
        extra["response_format"] = {"type": "json_object"}
    async for piece in _map_reduce_stream(
        document_text,
        map_prompt="Extract the key points from the following text in concise bullet points:",
        reduce_prompt="Based on these extracted key points from different sections of a document, write a 3-sentence professional summary:",
        single_chunk_prompt="Please provide a 3-sentence professional summary of the following document:",
        app_name=app_name,
        spec=spec,
        extra=extra,
    ):
        yield piece


async def generate_comparison(
    doc1_text: str, doc2_text: str, file_1: str, file_2: str, app_name: str, spec: ModelSpec,
    response_format: str = "text",
) -> AsyncGenerator[str, None]:
    """Compares two documents using map-reduce to preserve full context, then
    streams the final comparison. Both documents share one map fan-out so
    progress ticks cover the whole phase; the per-document digests are
    intermediate and stay non-streamed. ``response_format`` applies only to the
    final call."""
    extra: dict = {}
    if response_format == "json":
        extra["response_format"] = {"type": "json_object"}

    docs = [
        {"filename": filename, "chunks": chunk_text(text=text, max_words_per_chunk=1500), "start": 0}
        for filename, text in ((file_1, doc1_text), (file_2, doc2_text))
    ]

    # Combined map phase across both documents. A single-chunk document skips
    # map-reduce entirely — the chunk itself is its digest.
    jobs: list[list[dict[str, str]]] = []
    for doc in docs:
        if len(doc["chunks"]) == 1:
            continue
        doc["start"] = len(jobs)
        jobs += [
            [{"role": "user", "content": (
                f"Extract every distinct fact, rule, figure, and clause from the following excerpt of "
                f"'{doc['filename']}'. Be exhaustive — nothing should be lost. Use concise bullet points:\n\n{chunk}"
            )}]
            for chunk in doc["chunks"]
        ]

    extracted: list[str] = []
    if jobs:
        total = len(jobs)
        yield f"[PROGRESS:mapping {total} chunks]\n"
        async for item in map_calls_iter(jobs, app_name, spec):
            if isinstance(item, list):
                extracted = item
            else:
                yield f"[PROGRESS:mapped {item}/{total} chunks]\n"

    # Reduce each mapped document's notes into its digest, concurrently.
    async def _digest(doc: dict) -> str:
        if len(doc["chunks"]) == 1:
            return doc["chunks"][0]
        notes = extracted[doc["start"]: doc["start"] + len(doc["chunks"])]
        return await call_llm(
            [{"role": "user", "content": (
                f"The following are extracted notes from all sections of '{doc['filename']}'. Consolidate them "
                f"into a single, organised list of key facts — remove duplicates but preserve all unique "
                f"information:\n\n" + "\n\n".join(notes)
            )}],
            app_name,
            spec,
        )

    if jobs:
        yield "[PROGRESS:consolidating notes]\n"
    digest_1, digest_2 = await asyncio.gather(_digest(docs[0]), _digest(docs[1]))

    yield "[PROGRESS:comparing]\n"
    compare_messages = [
        {
            "role": "user",
            "content": (
                f"Compare these two documents. Provide a bulleted list of strictly what has changed "
                f"or what is distinctly different between them.\n\n"
                f"--- Document 1 ({file_1}) ---\n{digest_1}\n\n"
                f"--- Document 2 ({file_2}) ---\n{digest_2}"
            ),
        }
    ]
    async for token in stream_llm(compare_messages, app_name, spec, extra=extra):
        yield token
