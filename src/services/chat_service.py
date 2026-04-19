import os
from typing import AsyncGenerator
from src.utils.ai_client import get_async_ai_client, record_llm_usage
from src.utils.file_parser import chunk_text
from src.utils.memory import add_message, get_or_create_session
from src.utils.logger import logger
from src.utils.concurrency import gpu_slot

_client = get_async_ai_client()
_MODEL_NAME = os.getenv("MODEL_NAME", "gemma-api-test")


async def generate_chat_stream(
    prompt: str,
    app_name: str,
    system_prompt: str | None = None,
    session_id: str | None = None,
    response_format: str = "text",
) -> AsyncGenerator[str, None]:
    """Streams the chat response token-by-token."""
    active_session_id, history = await get_or_create_session(
        session_id=session_id,
        system_prompt=system_prompt,
        app_name=app_name,
    )

    await add_message(session_id=active_session_id, role="user", content=prompt, app_name=app_name)
    history.append({"role": "user", "content": prompt})

    extra: dict = {}
    if response_format == "json":
        extra["response_format"] = {"type": "json_object"}

    response = await _client.chat.completions.create(  # type: ignore[call-overload]
        model=_MODEL_NAME,
        messages=history,  # type: ignore[arg-type]
        stream=True,
        stream_options={"include_usage": True},
        **extra,
    )

    yield f"[SESSION_ID:{active_session_id}]\n"

    full_ai_response = ""
    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            word = chunk.choices[0].delta.content
            full_ai_response += word
            yield word
        if getattr(chunk, "usage", None):
            await record_llm_usage(chunk, app_name)

    await add_message(session_id=active_session_id, role="assistant", content=full_ai_response, app_name=app_name)


async def generate_file_summary(
    document_text: str,
    task: str,
    tone: str,
    app_name: str,
) -> AsyncGenerator[str, None]:
    """Processes a document based on user instructions, streaming progress events then the result."""
    text_chunks = chunk_text(text=document_text, max_words_per_chunk=1500)
    system_setup = f"You are a highly capable AI. Your tone must be: {tone}."
    total = len(text_chunks)

    if total == 1:
        prompt = f"{system_setup}\n\nTask: {task}"
        async for token in _stream_single_block(text_chunks[0], prompt, app_name):
            yield token
        return

    # MAP PHASE
    map_prompt = (
        f"{system_setup}\n\n"
        f"Task: Extract the information from the following text necessary to ultimately accomplish this goal: '{task}'."
    )
    mini_results = []
    for index, chunk in enumerate(text_chunks):
        logger.info(f"Processing chunk {index + 1} of {total}...")
        yield f"[PROGRESS:chunk {index + 1}/{total}]\n"
        mini_results.append(await _process_single_block(chunk, map_prompt, app_name))

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
    async for token in _stream_single_block(combined_text, reduce_prompt, app_name):
        yield token


async def _stream_single_block(
    text: str,
    system_instruction: str,
    app_name: str,
) -> AsyncGenerator[str, None]:
    """Streams the LLM response for a single block. Holds the GPU slot for the full stream duration."""
    async with gpu_slot():
        response = await _client.chat.completions.create(  # type: ignore[call-overload]
            model=_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": text},
            ],
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
            if getattr(chunk, "usage", None):
                await record_llm_usage(chunk, app_name)


async def _process_single_block(text: str, system_instruction: str, app_name: str) -> str:
    """Non-streaming LLM call for a single block. Used during the MAP phase."""
    async with gpu_slot():
        response = await _client.chat.completions.create(
            model=_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": text},
            ],
        )
    await record_llm_usage(response, app_name)
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("LLM returned no content for a text chunk.")
    return content
