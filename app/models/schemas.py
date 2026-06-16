from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    content_type: str | None = None
    chunks_count: int
    size_bytes: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_count: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: str | None = None


class Source(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    text_preview: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class DocumentsResponse(BaseModel):
    documents: list[DocumentMetadata]


class DeleteDocumentResponse(BaseModel):
    document_id: str
    deleted: bool


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    vector_store: str
    llm_mode: str
    embeddings_mode: str
    extra: dict[str, Any] = Field(default_factory=dict)

