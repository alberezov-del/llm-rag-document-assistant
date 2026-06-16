from uuid import uuid4

from app.core.config import Settings
from app.models.schemas import (
    AskResponse,
    ChunkRecord,
    DocumentUploadResponse,
    Source,
)
from app.services.chunker import TextChunker
from app.services.document_loader import extract_text_from_bytes
from app.services.embeddings import (
    EmbeddingProvider,
    MockEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from app.services.llm_client import LLMClient, MockLLMClient, OpenAICompatibleLLMClient
from app.services.vector_store import ChromaVectorStore, RetrievedChunk
from app.storage.document_store import DocumentStore


class RAGPipeline:
    """Coordinates document ingestion, retrieval, prompt building, and LLM calls."""

    def __init__(
        self,
        settings: Settings,
        document_store: DocumentStore | None = None,
        vector_store: ChromaVectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        llm_client: LLMClient | None = None,
        chunker: TextChunker | None = None,
    ) -> None:
        self.settings = settings
        self.document_store = document_store or DocumentStore(settings.data_dir)
        self.vector_store = vector_store or ChromaVectorStore(
            persist_dir=str(settings.resolved_chroma_dir),
            collection_name=settings.chroma_collection,
        )
        self.embedding_provider = embedding_provider or self._build_embedding_provider(settings)
        self.llm_client = llm_client or self._build_llm_client(settings)
        self.chunker = chunker or TextChunker(settings.chunk_size, settings.chunk_overlap)

    def ingest_file(
        self,
        filename: str,
        content: bytes,
        content_type: str | None = None,
    ) -> DocumentUploadResponse:
        text = extract_text_from_bytes(filename, content)
        raw_chunks = self.chunker.split(text)
        if not raw_chunks:
            raise ValueError("Document did not produce any chunks")

        document_id = str(uuid4())
        safe_filename = filename.rsplit("/", maxsplit=1)[-1].rsplit("\\", maxsplit=1)[-1]
        chunks = [
            ChunkRecord(
                chunk_id=f"{document_id}:{chunk.index:04d}",
                document_id=document_id,
                filename=safe_filename,
                chunk_index=chunk.index,
                text=chunk.text,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
            )
            for chunk in raw_chunks
        ]
        embeddings = self.embedding_provider.embed_texts([chunk.text for chunk in chunks])
        metadata = self.document_store.save_document(
            document_id=document_id,
            filename=safe_filename,
            content_type=content_type,
            raw_content=content,
            extracted_text=text,
            chunks=chunks,
        )
        self.vector_store.add_chunks(metadata, chunks, embeddings)

        return DocumentUploadResponse(
            document_id=metadata.document_id,
            filename=metadata.filename,
            chunks_count=metadata.chunks_count,
        )

    def ask(self, question: str, top_k: int = 5, document_id: str | None = None) -> AskResponse:
        query_embedding = self.embedding_provider.embed_texts([question])[0]
        retrieved_chunks = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=document_id,
        )
        context = self._build_context(retrieved_chunks)
        answer = self.llm_client.generate_answer(question=question, context=context)

        return AskResponse(
            answer=answer,
            sources=[
                Source(
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    chunk_id=chunk.chunk_id,
                    text_preview=self._preview(chunk.text),
                    score=chunk.score,
                )
                for chunk in retrieved_chunks
            ],
        )

    def delete_document(self, document_id: str) -> bool:
        deleted = self.document_store.delete_document(document_id)
        self.vector_store.delete_document(document_id)
        return deleted

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return ""

        context_blocks = []
        for index, chunk in enumerate(chunks, start=1):
            context_blocks.append(
                "\n".join(
                    [
                        f"[Source {index}]",
                        f"document_id: {chunk.document_id}",
                        f"filename: {chunk.filename}",
                        f"chunk_id: {chunk.chunk_id}",
                        f"score: {chunk.score}",
                        "text:",
                        chunk.text,
                    ]
                )
            )
        return "\n\n".join(context_blocks)

    def _preview(self, text: str) -> str:
        compact = " ".join(text.split())
        if len(compact) <= self.settings.max_preview_chars:
            return compact
        return compact[: self.settings.max_preview_chars].rstrip() + "..."

    def _build_embedding_provider(self, settings: Settings) -> EmbeddingProvider:
        if settings.should_use_mock_embeddings:
            return MockEmbeddingProvider()
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when mock embeddings are disabled")
        return OpenAICompatibleEmbeddingProvider(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.embedding_model,
            timeout_seconds=settings.request_timeout_seconds,
        )

    def _build_llm_client(self, settings: Settings) -> LLMClient:
        if settings.should_use_mock_llm:
            return MockLLMClient()
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when mock LLM is disabled")
        return OpenAICompatibleLLMClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.llm_model,
            timeout_seconds=settings.request_timeout_seconds,
        )

