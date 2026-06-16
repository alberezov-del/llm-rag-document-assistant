from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import Settings, get_settings
from app.models.schemas import (
    AskRequest,
    AskResponse,
    DeleteDocumentResponse,
    DocumentsResponse,
    DocumentUploadResponse,
    HealthResponse,
)
from app.services.rag_pipeline import RAGPipeline

router = APIRouter()


@lru_cache
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(get_settings())


SettingsDep = Annotated[Settings, Depends(get_settings)]
PipelineDep = Annotated[RAGPipeline, Depends(get_pipeline)]
UploadFileDep = Annotated[UploadFile, File(...)]


@router.get("/health", response_model=HealthResponse)
def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        vector_store="chromadb",
        llm_mode="mock" if settings.should_use_mock_llm else "openai-compatible",
        embeddings_mode="mock" if settings.should_use_mock_embeddings else "openai-compatible",
    )


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFileDep,
    pipeline: PipelineDep,
) -> DocumentUploadResponse:
    content = await file.read()
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename",
        )

    try:
        return pipeline.ingest_file(
            filename=file.filename,
            content=content,
            content_type=file.content_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    pipeline: PipelineDep,
) -> AskResponse:
    return pipeline.ask(
        question=request.question,
        top_k=request.top_k,
        document_id=request.document_id,
    )


@router.get("/documents", response_model=DocumentsResponse)
def list_documents(pipeline: PipelineDep) -> DocumentsResponse:
    return DocumentsResponse(documents=pipeline.document_store.list_documents())


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(
    document_id: str,
    pipeline: PipelineDep,
) -> DeleteDocumentResponse:
    deleted = pipeline.delete_document(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' was not found",
        )
    return DeleteDocumentResponse(document_id=document_id, deleted=True)
