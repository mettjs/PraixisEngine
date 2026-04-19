from pydantic import BaseModel, Field
from typing import Literal

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    system_prompt: str | None = None
    session_id: str | None = None
    response_format: Literal["text", "json"] = "text"

class QuestionRequest(BaseModel):
    collection_name: str = Field(..., pattern=r"^[a-zA-Z0-9_-]{3,63}$")
    question: str = Field(..., min_length=1)
    session_id: str | None = None
    n_results: int = Field(default=5, ge=1, le=20)
    system_prompt: str | None = None
    metadata_filter: dict | None = None

class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1)