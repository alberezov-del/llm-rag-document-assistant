from dataclasses import dataclass
from typing import Any, cast

import chromadb
from chromadb.api.models.Collection import Collection

from app.models.schemas import ChunkRecord, DocumentMetadata


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float


class ChromaVectorStore:
    """Thin ChromaDB wrapper that stores precomputed embeddings."""

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        document: DocumentMetadata,
        chunks: list[ChunkRecord],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=cast(Any, embeddings),
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ],
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]:
        where = cast(Any, {"document_id": document_id}) if document_id else None
        results = cast(
            dict[str, Any],
            self.collection.query(
            query_embeddings=cast(Any, [query_embedding]),
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
            ),
        )

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for text, metadata, distance in zip(documents, metadatas, distances, strict=True):
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(metadata["chunk_id"]),
                    document_id=str(metadata["document_id"]),
                    filename=str(metadata["filename"]),
                    text=text,
                    score=round(score, 4),
                )
            )

        return retrieved

    def delete_document(self, document_id: str) -> None:
        existing = self.collection.get(where={"document_id": document_id})
        ids = existing.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
