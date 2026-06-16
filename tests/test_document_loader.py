import pytest

from app.services.document_loader import extract_text_from_bytes


def test_load_markdown_document() -> None:
    content = b"# RAG Notes\n\nRetrieval augmented generation uses documents as context."

    text = extract_text_from_bytes("notes.md", content)

    assert "RAG Notes" in text
    assert "documents as context" in text


def test_load_text_document() -> None:
    content = b"Plain text document about vector search."

    text = extract_text_from_bytes("document.txt", content)

    assert text == "Plain text document about vector search."


def test_reject_unsupported_file_type() -> None:
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text_from_bytes("image.png", b"not a document")
