from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from app.api.routes import get_pipeline
from app.core.config import Settings
from app.main import app
from app.services.rag_pipeline import RAGPipeline


def build_test_pipeline(tmp_path: Path, max_upload_bytes: int = 10_000) -> RAGPipeline:
    settings = Settings(
        data_dir=tmp_path / "data",
        chroma_dir=tmp_path / "chroma",
        chroma_collection=f"api_test_documents_{uuid4().hex}",
        chunk_size=260,
        chunk_overlap=40,
        max_upload_bytes=max_upload_bytes,
        use_mock_llm=True,
        use_mock_embeddings=True,
    )
    settings.ensure_directories()
    return RAGPipeline(settings=settings)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_upload_ask_list_and_delete_document(tmp_path: Path) -> None:
    pipeline = build_test_pipeline(tmp_path)
    app.dependency_overrides[get_pipeline] = lambda: pipeline

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            upload_response = await client.post(
                "/documents/upload",
                files={"file": ("policy.md", b"# Policy\n\nRAG answers should cite sources.")},
            )
            assert upload_response.status_code == 200
            document_id = upload_response.json()["document_id"]

            ask_response = await client.post(
                "/ask",
                json={
                    "question": "What should RAG answers cite?",
                    "top_k": 3,
                    "document_id": document_id,
                },
            )
            assert ask_response.status_code == 200
            assert ask_response.json()["sources"][0]["document_id"] == document_id

            list_response = await client.get("/documents")
            assert list_response.status_code == 200
            assert list_response.json()["documents"][0]["document_id"] == document_id

            delete_response = await client.delete(f"/documents/{document_id}")
            assert delete_response.status_code == 200
            assert delete_response.json() == {"document_id": document_id, "deleted": True}
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_upload_rejects_oversized_file(tmp_path: Path) -> None:
    pipeline = build_test_pipeline(tmp_path, max_upload_bytes=10)
    app.dependency_overrides[get_pipeline] = lambda: pipeline

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/documents/upload",
                files={
                    "file": ("large.md", b"This document is too large for the configured limit.")
                },
            )
            assert response.status_code == 413
            assert "Maximum size" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()
