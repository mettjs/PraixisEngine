import os
from typing import Generator
from src.utils.ai_client import get_ai_client
from src.utils.file_parser import chunk_text
from src.utils.memory import add_message, get_or_create_session
from src.utils.logger import logger
from src.utils.concurrency import gpu_slot

_client = get_ai_client()
_MODEL_NAME = os.getenv("MODEL_NAME", "gemma-api-test") # default to a test model if not set, but should be overridden in production

def generate_chat_stream(prompt: str, app_name: str, system_prompt: str | None = None, session_id: str | None = None) -> Generator[str, None, None]:
    """Streams the chat response word-by-word."""
    
    active_session_id, history = get_or_create_session(session_id=session_id, 
                                                       system_prompt=system_prompt,
                                                       app_name=app_name)
    
    add_message(session_id=active_session_id, role="user", content=prompt, app_name=app_name)
    
    history.append({"role": "user", "content": prompt})
    
    response = _client.chat.completions.create(
        model=_MODEL_NAME,
        messages=history, # type: ignore[arg-type]
        stream=True
    )
    
    yield f"[SESSION_ID:{active_session_id}]\n"
    
    full_ai_response = ""
    
    for chunk in response:
        if chunk.choices[0].delta.content is not None: # type: ignore[union-attr]
            word = chunk.choices[0].delta.content      # type: ignore[union-attr]
            full_ai_response += word
            yield word
            
    add_message(session_id=active_session_id, role="assistant", content=full_ai_response, app_name=app_name)

def generate_file_summary(document_text: str, task: str, tone: str) -> str:
    """Dynamically processes a document based on user instructions."""
    text_chunks = chunk_text(text=document_text, max_words_per_chunk=1500)
    
    # Create the dynamic "System Identity"
    system_setup = f"You are a highly capable AI. Your tone must be: {tone}."

    if len(text_chunks) == 1:
        prompt = f"{system_setup}\n\nTask: {task}"
        return _process_single_block(text_chunks[0], prompt)

    # MAP PHASE
    mini_results = []
    for index, chunk in enumerate(text_chunks):
        logger.info(f"Processing chunk {index + 1} of {len(text_chunks)}...")
        map_prompt = (
            f"{system_setup}\n\n"
            f"Task: Extract the information from the following text necessary to ultimately accomplish this goal: '{task}'."
        )
        mini_results.append(_process_single_block(chunk, map_prompt))

    # REDUCE PHASE
    logger.info("Combining chunks for the final result...")
    combined_text = "\n\n".join(mini_results)
    
    reduce_prompt = (
        f"{system_setup}\n\n"
        f"The following text is a collection of extracted notes from a larger document.\n"
        f"Final Task: {task}\n"
        f"Use the notes to complete the final task perfectly."
    )
    
    return _process_single_block(combined_text, reduce_prompt)

def _process_single_block(text: str, system_instruction: str) -> str:
    with gpu_slot():
        response = _client.chat.completions.create(
            model=_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": text}
            ]
        )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("LLM returned no content for a text chunk.")
    return content
