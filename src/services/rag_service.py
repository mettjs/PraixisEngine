import os
from typing import Generator
from src.utils.ai_client import get_ai_client
from src.utils.file_parser import chunk_text
from src.utils.memory import add_message, get_or_create_session
from src.utils.concurrency import gpu_slot

_client = get_ai_client()
_MODEL_NAME = os.getenv("MODEL_NAME", "gemma-api-test") # default to a test model if not set, but should be overridden in production

def _call_llm(prompt: str) -> str:
    """Single non-streaming LLM call. Raises on empty response."""
    with gpu_slot():
        response = _client.chat.completions.create(
            model=_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
    content = response.choices[0].message.content  # type: ignore[union-attr]
    if content is None:
        raise RuntimeError("LLM returned no content.")
    return content

# -----------------------------------------------------------------------
# Function to generate an answer to a question based on context
# -----------------------------------------------------------------------
def generate_rag_answer(
    question: str,
    app_name: str,
    context_chunks: list[dict[str, str]],
    search_query: str,
    system_prompt: str | None = None,
    session_id: str | None = None
) -> Generator[str, None, None]:
    """Generates an answer to a question based only on the context provided by the collction."""
    if not system_prompt:
        system_prompt = (
            "You are an expert institutional analyst. "
            "Use ONLY the provided context to answer the user's question. "
            "If the answer is not contained in the context, explain that the answer cannot be found in the document, do not fabricate any information."
        )

    active_session_id, history = get_or_create_session(
        session_id=session_id,
        system_prompt=system_prompt,
        app_name=app_name
    )

    add_message(session_id=active_session_id, role="user", content=question, app_name=app_name)

    formatted_chunks = [f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk in context_chunks]
    context_text = "\n\n---\n\n".join(formatted_chunks)
    augmented_question = f"Context:\n{context_text}\n\nQuestion: {question}"

    temp_history = history.copy()
    temp_history.append({"role": "user", "content": augmented_question})

    # Metadata lines that clients can read these before the streamed answer begins
    yield f"[SESSION_ID:{active_session_id}]\n"
    yield f"[SEARCH_QUERY:{search_query}]\n"
    unique_sources = list({chunk["source"] for chunk in context_chunks})
    yield f"[SOURCES:{','.join(unique_sources)}]\n"

    response = _client.chat.completions.create(
        model=_MODEL_NAME,
        messages=temp_history, # type: ignore[arg-type]
        stream=True
    )

    full_answer = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None: # type: ignore[union-attr]
            token = chunk.choices[0].delta.content     # type: ignore[union-attr]
            full_answer += token
            yield token

    add_message(session_id=active_session_id, role="assistant", content=full_answer, app_name=app_name)

def reformulate_query(history: list, latest_question: str) -> str:
    """Uses the chat history to rewrite vague follow-up questions into standalone queries."""
    
    # If this is the very first question (only the system prompt exists in history) 
    # Return the question.
    if len(history) <= 1:
        return latest_question
        
    reformulation_prompt = (
        "Given the following conversation history and the user's latest question, "
        "rewrite the question to be a fully standalone search query. "
        "Do not answer the question, ONLY return the rewritten query."
    )
    
    # Extract just the readable text from the history to show the AI
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[1:]])
    user_msg = f"History:\n{history_text}\n\nLatest Question: {latest_question}"
    
    with gpu_slot():
        response = _client.chat.completions.create(
            model=_MODEL_NAME,
            messages=[
                {"role": "system", "content": reformulation_prompt},
                {"role": "user", "content": user_msg}
            ]
        )

    # Extract the rewritten query (or fallback to the original if something fails)
    content = response.choices[0].message.content  # type: ignore[union-attr]
    return content.strip() if content else latest_question

def _map_reduce(
    text: str,
    map_prompt: str,
    reduce_prompt: str,
    single_chunk_prompt: str | None = None
) -> str:
    """Runs a map-reduce pipeline over chunked text using the LLM.

    For single-chunk documents, calls the LLM with `single_chunk_prompt` if provided,
    or returns the raw text if not (useful when the caller will process it further).
    """
    chunks = chunk_text(text=text, max_words_per_chunk=1500)

    if len(chunks) == 1:
        return _call_llm(f"{single_chunk_prompt}\n\n{chunks[0]}") if single_chunk_prompt else chunks[0]

    extracted = [_call_llm(f"{map_prompt}\n\n{chunk}") for chunk in chunks]
    return _call_llm(f"{reduce_prompt}\n\n" + "\n\n".join(extracted))

def generate_summary(document_text: str) -> str:
    """Summarizes a document using map-reduce for large texts."""
    return _map_reduce(
        document_text,
        map_prompt="Extract the key points from the following text in concise bullet points:",
        reduce_prompt="Based on these extracted key points from different sections of a document, write a 3-sentence professional summary:",
        single_chunk_prompt="Please provide a 3-sentence professional summary of the following document:",
    )
    
def generate_comparison(doc1_text: str, doc2_text: str, file_1: str, file_2: str) -> str:
    """Compares two documents using map-reduce to preserve full context."""
    digest_1 = _map_reduce(
        doc1_text,
        map_prompt=f"Extract every distinct fact, rule, figure, and clause from the following excerpt of '{file_1}'. Be exhaustive — nothing should be lost. Use concise bullet points:",
        reduce_prompt=f"The following are extracted notes from all sections of '{file_1}'. Consolidate them into a single, organised list of key facts — remove duplicates but preserve all unique information:",
    )
    digest_2 = _map_reduce(
        doc2_text,
        map_prompt=f"Extract every distinct fact, rule, figure, and clause from the following excerpt of '{file_2}'. Be exhaustive — nothing should be lost. Use concise bullet points:",
        reduce_prompt=f"The following are extracted notes from all sections of '{file_2}'. Consolidate them into a single, organised list of key facts — remove duplicates but preserve all unique information:",
    )

    return _call_llm(
        f"Compare these two documents. Provide a bulleted list of strictly what has changed "
        f"or what is distinctly different between them.\n\n"
        f"--- Document 1 ({file_1}) ---\n{digest_1}\n\n"
        f"--- Document 2 ({file_2}) ---\n{digest_2}"
    )