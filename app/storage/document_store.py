import json
import shutil
from pathlib import Path
from typing import Any, cast

from app.models.schemas import ChunkRecord, DocumentMetadata


class DocumentStore:
    """JSON-based local metadata and chunk store."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.uploads_dir = self.data_dir / "uploads"
        self.documents_dir = self.data_dir / "documents"
        self.index_path = self.data_dir / "documents.json"

        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._write_index({})

    def save_document(
        self,
        document_id: str,
        filename: str,
        content_type: str | None,
        raw_content: bytes,
        extracted_text: str,
        chunks: list[ChunkRecord],
    ) -> DocumentMetadata:
        safe_filename = Path(filename).name
        upload_dir = self.uploads_dir / document_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_path = upload_dir / safe_filename
        upload_path.write_bytes(raw_content)

        metadata = DocumentMetadata(
            document_id=document_id,
            filename=safe_filename,
            content_type=content_type,
            chunks_count=len(chunks),
            size_bytes=len(raw_content),
        )
        payload = {
            "document": metadata.model_dump(mode="json"),
            "upload_path": str(upload_path),
            "extracted_text": extracted_text,
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        }
        (self.documents_dir / f"{document_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        index = self._read_index()
        index[document_id] = metadata.model_dump(mode="json")
        self._write_index(index)
        return metadata

    def list_documents(self) -> list[DocumentMetadata]:
        index = self._read_index()
        return [
            DocumentMetadata.model_validate(value)
            for value in sorted(index.values(), key=lambda item: item["created_at"], reverse=True)
        ]

    def get_document(self, document_id: str) -> DocumentMetadata | None:
        metadata = self._read_index().get(document_id)
        if metadata is None:
            return None
        return DocumentMetadata.model_validate(metadata)

    def delete_document(self, document_id: str) -> bool:
        index = self._read_index()
        if document_id not in index:
            return False

        index.pop(document_id)
        self._write_index(index)

        document_path = self.documents_dir / f"{document_id}.json"
        if document_path.exists():
            document_path.unlink()

        upload_dir = self.uploads_dir / document_id
        if upload_dir.exists():
            shutil.rmtree(upload_dir)

        return True

    def _read_index(self) -> dict[str, dict[str, Any]]:
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Document index is corrupted")
        return cast(dict[str, dict[str, Any]], payload)

    def _write_index(self, index: dict[str, dict[str, Any]]) -> None:
        self.index_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
