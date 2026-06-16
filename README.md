# LLM RAG Document Assistant

[![CI](https://github.com/alberezov-del/llm-rag-document-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/alberezov-del/llm-rag-document-assistant/actions/workflows/ci.yml)

FastAPI service that turns PDF, TXT, and Markdown files into a local RAG knowledge base.
Users upload a document, the service extracts text, splits it into chunks, creates embeddings,
stores them in ChromaDB, and answers questions with cited source snippets.

The project is designed as a portfolio-ready Junior AI Engineer / LLM Engineer project:
it shows API design, document ingestion, vector search, provider abstraction, testing,
Docker packaging, and honest local demo behavior without real API keys.

## Why This Project Is Useful for a Junior AI Engineer

- It demonstrates the full RAG lifecycle rather than only calling an LLM API.
- It separates document loading, chunking, embeddings, vector search, prompting, and LLM calls.
- It works without paid credentials through deterministic mock providers.
- It is testable: core logic is covered by pytest and does not require API keys.
- It is production-shaped: FastAPI, Pydantic schemas, Docker, GitHub Actions CI,
  `.env.example`, and lint tooling.

## Stack

- Python 3.11+
- FastAPI
- Pydantic
- ChromaDB
- pypdf
- OpenAI-compatible LLM and embeddings clients
- python-dotenv
- Docker and docker-compose
- pytest
- ruff
- mypy

## Architecture

```text
Client
  -> FastAPI routes
  -> RAGPipeline
      -> document_loader: PDF/TXT/Markdown text extraction
      -> chunker: overlapping text chunks
      -> embeddings: mock or OpenAI-compatible vectors
      -> vector_store: ChromaDB persistence and retrieval
      -> document_store: local JSON metadata and uploaded files
      -> llm_client: mock or OpenAI-compatible chat completion
  -> JSON answer with cited sources
```

Detailed architecture is documented in [docs/architecture.md](docs/architecture.md).

## Local Setup

```bash
git clone https://github.com/alberezov-del/llm-rag-document-assistant.git
cd llm-rag-document-assistant

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

cp .env.example .env
uvicorn app.main:app --reload
```

Open:

- API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

By default the example `.env` uses mock LLM and mock embeddings, so no paid API key is
required for a local demo.

## Docker Setup

```bash
docker compose up --build
```

The API will be available at:

```text
http://127.0.0.1:8000
```

The compose setup uses mock providers by default and stores persistent local data in a
Docker volume named `rag_data`.

## API Examples

### Upload a Document

```bash
curl -X POST "http://127.0.0.1:8000/documents/upload" \
  -F "file=@sample_docs/sample_ai_policy.md"
```

Example response:

```json
{
  "document_id": "9d79e72c-8df0-4e3b-b3b4-c780f4a6dcd4",
  "filename": "sample_ai_policy.md",
  "chunks_count": 2
}
```

### Ask a Question

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Why should RAG systems return source snippets?",
    "top_k": 5
  }'
```

Question limited to one document:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What must employees avoid uploading?",
    "top_k": 3,
    "document_id": "9d79e72c-8df0-4e3b-b3b4-c780f4a6dcd4"
  }'
```

Example response:

```json
{
  "answer": "Mock answer: based on the retrieved document context, the answer is likely related to: [Source 1] document_id: ...",
  "sources": [
    {
      "document_id": "9d79e72c-8df0-4e3b-b3b4-c780f4a6dcd4",
      "filename": "sample_ai_policy.md",
      "chunk_id": "9d79e72c-8df0-4e3b-b3b4-c780f4a6dcd4:0000",
      "text_preview": "Acme Research Lab uses AI assistants to help employees summarize internal documents...",
      "score": 0.83
    }
  ]
}
```

### List Documents

```bash
curl "http://127.0.0.1:8000/documents"
```

### Delete a Document

```bash
curl -X DELETE "http://127.0.0.1:8000/documents/<document_id>"
```

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `APP_NAME` | `LLM RAG Document Assistant` | API title. |
| `APP_VERSION` | `0.1.0` | API version. |
| `DATA_DIR` | `data` | Local metadata and upload storage directory. |
| `CHROMA_DIR` | `data/chroma` | ChromaDB persistence directory. |
| `CHROMA_COLLECTION` | `documents` | ChromaDB collection name. |
| `CHUNK_SIZE` | `1000` | Target chunk size in characters. |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks. |
| `MAX_PREVIEW_CHARS` | `240` | Maximum source preview length in API responses. |
| `MAX_UPLOAD_BYTES` | `10485760` | Maximum accepted upload size in bytes. |
| `OPENAI_API_KEY` | empty | API key for OpenAI-compatible providers. |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Provider base URL. |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model name. |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name. |
| `USE_MOCK_LLM` | `true` in `.env.example` | Use local mock LLM instead of external API. |
| `USE_MOCK_EMBEDDINGS` | `true` in `.env.example` | Use deterministic local embeddings. |
| `REQUEST_TIMEOUT_SECONDS` | `60` | HTTP timeout for provider requests. |
| `LOG_LEVEL` | `INFO` | Application log level. |

## How RAG Works in This Project

1. The user uploads a PDF, TXT, or Markdown file.
2. The loader extracts plain text.
3. The chunker splits text into overlapping chunks and keeps offsets.
4. The embedding provider turns chunks into vectors.
5. ChromaDB stores vectors with chunk text and source metadata.
6. The user asks a question.
7. The same embedding provider embeds the question.
8. ChromaDB retrieves the most similar chunks.
9. The pipeline builds a context prompt with source labels.
10. The LLM client generates an answer from the retrieved context.
11. The API returns the answer plus source previews and similarity scores.

## Mock Mode

Mock mode is intentionally included for local demonstration without paid credentials:

- `MockEmbeddingProvider` creates deterministic hash-based vectors.
- `MockLLMClient` generates a simple answer from retrieved context.

This is not a replacement for real embeddings or a real LLM. It makes the project runnable
in CI, during interviews, and on a fresh laptop without secrets.

To use a real provider:

```env
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
USE_MOCK_LLM=false
USE_MOCK_EMBEDDINGS=false
```

Any provider that supports OpenAI-compatible `/embeddings` and `/chat/completions` endpoints
can be wired through the same abstraction.

## Tests and Quality Checks

```bash
pytest
ruff check .
mypy app
```

Tests run without API keys.

## License

MIT License. See [LICENSE](LICENSE).

## Limitations and Future Improvements

- Mock embeddings are deterministic but not semantically strong.
- PDF extraction depends on embedded text; scanned PDFs need OCR.
- There is no authentication or user-level document isolation yet.
- ChromaDB is local-only in this project.
- Chunking is character-based; token-aware chunking would be better for production.
- Future improvements could add streaming answers, reranking, OCR, async ingestion jobs,
  user accounts, rate limiting, and an evaluation dataset for retrieval quality.

## What I Learned

- How to design a RAG pipeline as composable services.
- How to separate external provider integrations from core business logic.
- How to make LLM projects reproducible without storing secrets.
- How to return cited sources instead of opaque LLM answers.
- How to test RAG behavior in mock mode without depending on network calls.
