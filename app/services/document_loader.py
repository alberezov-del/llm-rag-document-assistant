from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


def extract_text_from_bytes(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{extension}'. Supported types: {supported}")

    if extension == ".pdf":
        return _extract_pdf_text(content)

    return _decode_text(content)


def _extract_pdf_text(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"[Page {page_number}]\n{page_text.strip()}")

    text = "\n\n".join(pages).strip()
    if not text:
        raise ValueError("No extractable text found in PDF")
    return text


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Could not decode text file")

    text = text.strip()
    if not text:
        raise ValueError("The uploaded document is empty")
    return text
