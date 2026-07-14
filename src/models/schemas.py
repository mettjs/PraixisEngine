from pydantic import BaseModel, Field
from typing import Literal

from src.config import MAX_FILE_SIZE

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    system_prompt: str | None = None
    session_id: str | None = None
    stream: bool = True
    response_format: Literal["text", "json"] = "text"

class QuestionRequest(BaseModel):
    collection_name: str = Field(..., pattern=r"^[a-zA-Z0-9_-]{3,63}$")
    question: str = Field(..., min_length=1)
    session_id: str | None = None
    n_results: int = Field(default=5, ge=1, le=20)
    system_prompt: str | None = None
    metadata_filter: dict | None = None
    stream: bool = True
    response_format: Literal["text", "json"] = "text"

class TextUploadRequest(BaseModel):
    # The same cap enforced on file uploads (chars here, bytes there — close enough).
    text: str = Field(..., min_length=1, max_length=MAX_FILE_SIZE)
    filename: str = Field(..., min_length=1, max_length=255,
                          description="Stored identity of the document, e.g. 'faq-2026.txt'.")
    collection_name: str = Field(default="main", pattern=r"^[a-zA-Z0-9_-]{3,63}$")
    chunk_size: int = Field(default=2000, ge=100, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=500)
    chunking_strategy: Literal["semantic", "character"] = "semantic"
    improved_search: bool = False

class SearchRequest(BaseModel):
    collection_name: str = Field(..., pattern=r"^[a-zA-Z0-9_-]{3,63}$")
    query: str = Field(..., min_length=1)
    n_results: int = Field(default=5, ge=1, le=20)

class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1)

class CompareRequest(BaseModel):
    collection_name: str = Field(..., pattern=r"^[a-zA-Z0-9_-]{3,63}$")
    file_1: str = Field(..., min_length=1)
    file_2: str = Field(..., min_length=1)
    stream: bool = False
    response_format: Literal["text", "json"] = "text"