import asyncio
from collections.abc import AsyncGenerator
from src.utils.file_parser import chunk_text
from src.services.llm_runner import call_llm, map_calls
from src.services.session_stream import open_user_turn, stream_assistant_turn


async def generate_rag_answer(
    question: str,
    app_name: str,
    context_chunks: list[dict[str, str]],
    search_query: str,
    system_prompt: str | None = None,
    session_id: str | None = None,
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
        session_id=session_id,
        system_prompt=system_prompt,
    )

    formatted_chunks = [f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk in context_chunks]
    context_text = "\n\n---\n\n".join(formatted_chunks)
    augmented_question = f"Context:\n{context_text}\n\nQuestion: {question}"

    # The LLM sees the augmented question; the session stores the original one.
    temp_history = history[:-1] + [{"role": "user", "content": augmented_question}]

    yield f"[SESSION_ID:{active_session_id}]\n"
    yield f"[SEARCH_QUERY:{search_query}]\n"
    unique_sources = list({chunk["source"] for chunk in context_chunks})
    yield f"[SOURCES:{','.join(unique_sources)}]\n"

    async for token in stream_assistant_turn(
        messages=temp_history,
        app_name=app_name,
        session_id=active_session_id,
        history=history,
    ):
        yield token


async def reformulate_query(history: list, latest_question: str, app_name: str) -> str:
    """Uses recent chat history to rewrite follow-up questions into standalone queries.

    Returns the question unchanged when it introduces a new, independent topic so that
    stale context from prior exchanges does not pollute an unrelated search query.
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
    )
    return content.strip() if content else latest_question


async def _map_reduce(
    text: str,
    map_prompt: str,
    reduce_prompt: str,
    app_name: str,
    single_chunk_prompt: str | None = None,
) -> str:
    """Runs a map-reduce pipeline over chunked text using the LLM."""
    chunks = chunk_text(text=text, max_words_per_chunk=1500)

    if len(chunks) == 1:
        if single_chunk_prompt:
            return await call_llm(
                [{"role": "user", "content": f"{single_chunk_prompt}\n\n{chunks[0]}"}], app_name
            )
        return chunks[0]

    extracted = await map_calls(
        [[{"role": "user", "content": f"{map_prompt}\n\n{chunk}"}] for chunk in chunks],
        app_name,
    )
    return await call_llm(
        [{"role": "user", "content": f"{reduce_prompt}\n\n" + "\n\n".join(extracted)}], app_name
    )


async def generate_summary(document_text: str, app_name: str) -> str:
    """Summarizes a document using map-reduce for large texts."""
    return await _map_reduce(
        document_text,
        map_prompt="Extract the key points from the following text in concise bullet points:",
        reduce_prompt="Based on these extracted key points from different sections of a document, write a 3-sentence professional summary:",
        app_name=app_name,
        single_chunk_prompt="Please provide a 3-sentence professional summary of the following document:",
    )


async def generate_comparison(doc1_text: str, doc2_text: str, file_1: str, file_2: str, app_name: str) -> str:
    """Compares two documents using map-reduce to preserve full context."""
    digest_1, digest_2 = await asyncio.gather(
        _map_reduce(
            doc1_text,
            map_prompt=f"Extract every distinct fact, rule, figure, and clause from the following excerpt of '{file_1}'. Be exhaustive — nothing should be lost. Use concise bullet points:",
            reduce_prompt=f"The following are extracted notes from all sections of '{file_1}'. Consolidate them into a single, organised list of key facts — remove duplicates but preserve all unique information:",
            app_name=app_name,
        ),
        _map_reduce(
            doc2_text,
            map_prompt=f"Extract every distinct fact, rule, figure, and clause from the following excerpt of '{file_2}'. Be exhaustive — nothing should be lost. Use concise bullet points:",
            reduce_prompt=f"The following are extracted notes from all sections of '{file_2}'. Consolidate them into a single, organised list of key facts — remove duplicates but preserve all unique information:",
            app_name=app_name,
        ),
    )

    return await call_llm(
        [
            {
                "role": "user",
                "content": (
                    f"Compare these two documents. Provide a bulleted list of strictly what has changed "
                    f"or what is distinctly different between them.\n\n"
                    f"--- Document 1 ({file_1}) ---\n{digest_1}\n\n"
                    f"--- Document 2 ({file_2}) ---\n{digest_2}"
                ),
            }
        ],
        app_name,
    )
