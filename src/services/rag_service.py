import os
from typing import AsyncGenerator
from src.utils.ai_client import get_async_ai_client, record_llm_usage
from src.utils.file_parser import chunk_text
from src.utils.memory import add_message, get_or_create_session
from src.utils.concurrency import gpu_slot

_client = get_async_ai_client()
_MODEL_NAME = os.getenv("MODEL_NAME", "gemma-api-test")


async def _call_llm(prompt: str, app_name: str) -> str:
    """Single non-streaming LLM call. Raises on empty response."""
    async with gpu_slot():
        response = await _client.chat.completions.create(
            model=_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
    await record_llm_usage(response, app_name)
    content = response.choices[0].message.content  # type: ignore[union-attr]
    if content is None:
        raise RuntimeError("LLM returned no content.")
    return content


async def generate_rag_answer(
    question: str,
    app_name: str,
    context_chunks: list[dict[str, str]],
    search_query: str,
    system_prompt: str | None = None,
    session_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Generates an answer to a question based only on the context provided by the collection."""
    if not system_prompt:
        system_prompt = (
            "You are an expert institutional analyst. "
            "Use ONLY the provided context to answer the user's question. "
            "If the answer is not contained in the context, explain that the answer cannot be found in the document, do not fabricate any information."
        )

    active_session_id, history = await get_or_create_session(
        session_id=session_id,
        system_prompt=system_prompt,
        app_name=app_name,
    )

    await add_message(session_id=active_session_id, role="user", content=question, app_name=app_name)

    formatted_chunks = [f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk in context_chunks]
    context_text = "\n\n---\n\n".join(formatted_chunks)
    augmented_question = f"Context:\n{context_text}\n\nQuestion: {question}"

    temp_history = history.copy()
    temp_history.append({"role": "user", "content": augmented_question})

    yield f"[SESSION_ID:{active_session_id}]\n"
    yield f"[SEARCH_QUERY:{search_query}]\n"
    unique_sources = list({chunk["source"] for chunk in context_chunks})
    yield f"[SOURCES:{','.join(unique_sources)}]\n"

    response = await _client.chat.completions.create(  # type: ignore[call-overload, arg-type]
        model=_MODEL_NAME,
        messages=temp_history,
        stream=True,
        stream_options={"include_usage": True},
    )

    full_answer = ""
    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            full_answer += token
            yield token
        if getattr(chunk, "usage", None):
            await record_llm_usage(chunk, app_name)

    await add_message(session_id=active_session_id, role="assistant", content=full_answer, app_name=app_name)


async def reformulate_query(history: list, latest_question: str, app_name: str) -> str:
    """Uses the chat history to rewrite vague follow-up questions into standalone queries."""
    if len(history) <= 1:
        return latest_question

    reformulation_prompt = (
        "Given the following conversation history and the user's latest question, "
        "rewrite the question to be a fully standalone search query. "
        "Do not answer the question, ONLY return the rewritten query."
    )

    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[1:]])
    user_msg = f"History:\n{history_text}\n\nLatest Question: {latest_question}"

    async with gpu_slot():
        response = await _client.chat.completions.create(
            model=_MODEL_NAME,
            messages=[
                {"role": "system", "content": reformulation_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
    await record_llm_usage(response, app_name)

    content = response.choices[0].message.content  # type: ignore[union-attr]
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
            return await _call_llm(f"{single_chunk_prompt}\n\n{chunks[0]}", app_name)
        return chunks[0]

    extracted = []
    for chunk in chunks:
        extracted.append(await _call_llm(f"{map_prompt}\n\n{chunk}", app_name))
    return await _call_llm(f"{reduce_prompt}\n\n" + "\n\n".join(extracted), app_name)


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
    digest_1 = await _map_reduce(
        doc1_text,
        map_prompt=f"Extract every distinct fact, rule, figure, and clause from the following excerpt of '{file_1}'. Be exhaustive — nothing should be lost. Use concise bullet points:",
        reduce_prompt=f"The following are extracted notes from all sections of '{file_1}'. Consolidate them into a single, organised list of key facts — remove duplicates but preserve all unique information:",
        app_name=app_name,
    )
    digest_2 = await _map_reduce(
        doc2_text,
        map_prompt=f"Extract every distinct fact, rule, figure, and clause from the following excerpt of '{file_2}'. Be exhaustive — nothing should be lost. Use concise bullet points:",
        reduce_prompt=f"The following are extracted notes from all sections of '{file_2}'. Consolidate them into a single, organised list of key facts — remove duplicates but preserve all unique information:",
        app_name=app_name,
    )

    return await _call_llm(
        f"Compare these two documents. Provide a bulleted list of strictly what has changed "
        f"or what is distinctly different between them.\n\n"
        f"--- Document 1 ({file_1}) ---\n{digest_1}\n\n"
        f"--- Document 2 ({file_2}) ---\n{digest_2}",
        app_name,
    )
