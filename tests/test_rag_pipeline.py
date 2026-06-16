from pathlib import Path
from uuid import uuid4

from app.core.config import Settings
from app.services.rag_pipeline import RAGPipeline


def test_rag_pipeline_ingests_and_answers_in_mock_mode(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        chroma_dir=tmp_path / "chroma",
        chroma_collection=f"test_documents_{uuid4().hex}",
        chunk_size=260,
        chunk_overlap=40,
        use_mock_llm=True,
        use_mock_embeddings=True,
    )
    settings.ensure_directories()
    pipeline = RAGPipeline(settings=settings)

    content = b"""
    # AI Governance

    Retrieval augmented generation should provide cited source snippets.
    Human reviewers must verify AI-generated answers before business decisions.
    Monitoring failed searches improves retrieval quality over time.
    """

    uploaded = pipeline.ingest_file("policy.md", content, "text/markdown")
    response = pipeline.ask(
        question="Why should RAG assistants return source snippets?",
        top_k=3,
        document_id=uploaded.document_id,
    )

    assert uploaded.filename == "policy.md"
    assert uploaded.chunks_count >= 1
    assert response.sources
    assert response.sources[0].document_id == uploaded.document_id
    assert "Mock answer" in response.answer
    assert "source" in response.answer.lower() or "retrieval" in response.answer.lower()

