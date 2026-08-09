from collections.abc import AsyncGenerator
from src.models.registry import ModelSpec
from src.utils.file_parser import chunk_text
from src.utils.system.logger import logger
from src.services.llm_runner import stream_llm, map_calls_iter
from src.services.session_stream import open_user_turn, stream_assistant_turn


async def generate_chat_stream(
    prompt: str,
    app_name: str,
    spec: ModelSpec,
    system_prompt: str | None = None,
    session_id: str | None = None,
    response_format: str = "text",
    model_was_explicit: bool = False,
) -> AsyncGenerator[str, None]:
    """Streams the chat response token-by-token, from the model in ``spec``.

    The GPU slot is acquired by the controller and released by the
    ``SlotReleasingStreamingResponse`` wrapping this generator.
    """
    active_session_id, history = await open_user_turn(
        app_name=app_name,
        user_message=prompt,
        spec=spec,
        session_id=session_id,
        system_prompt=system_prompt,
        model_was_explicit=model_was_explicit,
    )

    extra: dict = {}
    if response_format == "json":
        extra["response_format"] = {"type": "json_object"}

    yield f"[SESSION_ID:{active_session_id}]\n"
    yield f"[MODEL:{spec.id}]\n"
    async for token in stream_assistant_turn(
        messages=history,
        app_name=app_name,
        session_id=active_session_id,
        history=history,
        spec=spec,
        extra=extra,
    ):
        yield token


async def generate_file_summary(
    document_text: str,
    task: str,
    tone: str,
    app_name: str,
    spec: ModelSpec,
    response_format: str = "text",
) -> AsyncGenerator[str, None]:
    """Processes a document based on user instructions, streaming progress events
    then the result. Each LLM call manages its own GPU slot via the shared runner,
    so the caller must NOT pre-acquire a slot.

    ``response_format`` is applied only to the final synthesis call — the map
    phase produces intermediate notes that are concatenated, so it stays text."""
    text_chunks = chunk_text(text=document_text, max_words_per_chunk=1500)
    system_setup = f"You are a highly capable AI. Your tone must be: {tone}."
    total = len(text_chunks)

    extra: dict = {}
    if response_format == "json":
        extra["response_format"] = {"type": "json_object"}

    if total == 1:
        messages = [
            {"role": "system", "content": f"{system_setup}\n\nTask: {task}"},
            {"role": "user", "content": text_chunks[0]},
        ]
        async for token in stream_llm(messages, app_name, spec, extra=extra):
            yield token
        return

    # MAP PHASE — chunks run concurrently (bounded by CHUNK_CONCURRENCY)
    map_prompt = (
        f"{system_setup}\n\n"
        f"Task: Extract the information from the following text necessary to ultimately accomplish this goal: '{task}'."
    )
    logger.info(f"Mapping {total} chunks...")
    yield f"[PROGRESS:mapping {total} chunks]\n"
    mini_results: list[str] = []
    async for item in map_calls_iter(
        [
            [
                {"role": "system", "content": map_prompt},
                {"role": "user", "content": chunk},
            ]
            for chunk in text_chunks
        ],
        app_name,
        spec,
    ):
        if isinstance(item, list):
            mini_results = item
        else:
            yield f"[PROGRESS:mapped {item}/{total} chunks]\n"

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
    async for token in stream_llm(reduce_messages, app_name, spec, extra=extra):
        yield token
