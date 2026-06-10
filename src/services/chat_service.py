from collections.abc import AsyncGenerator
from src.utils.file_parser import chunk_text
from src.utils.system.logger import logger
from src.services.llm_runner import stream_llm, map_calls
from src.services.session_stream import open_user_turn, stream_assistant_turn


async def generate_chat_stream(
    prompt: str,
    app_name: str,
    system_prompt: str | None = None,
    session_id: str | None = None,
    response_format: str = "text",
) -> AsyncGenerator[str, None]:
    """Streams the chat response token-by-token.

    The GPU slot is acquired by the controller and released by the
    ``SlotReleasingStreamingResponse`` wrapping this generator.
    """
    active_session_id, history = await open_user_turn(
        app_name=app_name,
        user_message=prompt,
        session_id=session_id,
        system_prompt=system_prompt,
    )

    extra: dict = {}
    if response_format == "json":
        extra["response_format"] = {"type": "json_object"}

    yield f"[SESSION_ID:{active_session_id}]\n"
    async for token in stream_assistant_turn(
        messages=history,
        app_name=app_name,
        session_id=active_session_id,
        history=history,
        extra=extra,
    ):
        yield token


async def generate_file_summary(
    document_text: str,
    task: str,
    tone: str,
    app_name: str,
) -> AsyncGenerator[str, None]:
    """Processes a document based on user instructions, streaming progress events
    then the result. Each LLM call manages its own GPU slot via the shared runner,
    so the caller must NOT pre-acquire a slot."""
    text_chunks = chunk_text(text=document_text, max_words_per_chunk=1500)
    system_setup = f"You are a highly capable AI. Your tone must be: {tone}."
    total = len(text_chunks)

    if total == 1:
        messages = [
            {"role": "system", "content": f"{system_setup}\n\nTask: {task}"},
            {"role": "user", "content": text_chunks[0]},
        ]
        async for token in stream_llm(messages, app_name):
            yield token
        return

    # MAP PHASE — chunks run concurrently (bounded by CHUNK_CONCURRENCY)
    map_prompt = (
        f"{system_setup}\n\n"
        f"Task: Extract the information from the following text necessary to ultimately accomplish this goal: '{task}'."
    )
    logger.info(f"Mapping {total} chunks...")
    yield f"[PROGRESS:mapping {total} chunks]\n"
    mini_results = await map_calls(
        [
            [
                {"role": "system", "content": map_prompt},
                {"role": "user", "content": chunk},
            ]
            for chunk in text_chunks
        ],
        app_name,
    )

    # REDUCE PHASE — stream the synthesised answer
    logger.info("Combining chunks for the final result...")
    yield f"[PROGRESS:reducing {total} chunks]\n"
    combined_text = "\n\n".join(mini_results)
    reduce_prompt = (
        f"{system_setup}\n\n"
        f"The following text is a collection of extracted notes from a larger document.\n"
        f"Final Task: {task}\n"
        f"Use the notes to complete the final task perfectly."
    )
    reduce_messages = [
        {"role": "system", "content": reduce_prompt},
        {"role": "user", "content": combined_text},
    ]
    async for token in stream_llm(reduce_messages, app_name):
        yield token
