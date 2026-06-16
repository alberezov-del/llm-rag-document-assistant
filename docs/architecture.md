# Architecture

This project implements a local document RAG pipeline with explicit, testable layers.

## Pipeline

```text
document upload
  -> text extraction
  -> chunking
  -> embeddings
  -> vector DB
  -> retrieval
  -> prompt building
  -> LLM answer
  -> cited sources
```

## 1. Document Upload

`POST /documents/upload` accepts PDF, TXT, Markdown, and `.markdown` files. The FastAPI
route reads the uploaded bytes and passes them to `RAGPipeline.ingest_file`.

## 2. Text Extraction

`app/services/document_loader.py` detects the file extension:

- PDF files are parsed with `pypdf.PdfReader`.
- TXT, Markdown, and `.markdown` files are decoded as text.
- Unsupported file types are rejected with a clear validation error.

The extracted text is stored locally together with the original upload metadata.

## 3. Chunking

`app/services/chunker.py` splits the extracted text into overlapping chunks. The chunker
prefers natural boundaries such as paragraphs, new lines, sentences, and spaces before
falling back to a hard character limit.

Each chunk receives:

- `chunk_id`
- `document_id`
- `filename`
- `chunk_index`
- `start_char`
- `end_char`
- chunk text

## 4. Embeddings

`app/services/embeddings.py` defines an `EmbeddingProvider` protocol and two implementations:

- `OpenAICompatibleEmbeddingProvider` calls an OpenAI-compatible `/embeddings` API.
- `MockEmbeddingProvider` creates deterministic local vectors for demos and tests.

The mock mode is not a semantic embedding model. It exists so the whole project can run
without paid credentials.

## 5. Vector Database

`app/services/vector_store.py` wraps ChromaDB. Chunks are stored with precomputed embeddings,
source text, and metadata. Retrieval queries can be filtered by `document_id` when the user
wants to ask questions about one document only.

## 6. Retrieval

`RAGPipeline.ask` embeds the question, queries ChromaDB, converts distances to similarity
scores, and builds a source-aware context block.

## 7. Prompt Building

The prompt contains one block per retrieved chunk:

```text
[Source 1]
document_id: ...
filename: ...
chunk_id: ...
score: ...
text:
...
```

This makes source attribution visible to the model and allows the API response to return
the same source metadata to the caller.

## 8. LLM Answer

`app/services/llm_client.py` defines an `LLMClient` protocol and two implementations:

- `OpenAICompatibleLLMClient` calls an OpenAI-compatible `/chat/completions` API.
- `MockLLMClient` produces deterministic local answers based on retrieved context.

## 9. Cited Sources

`POST /ask` returns an answer and a `sources` array. Each source contains:

- `document_id`
- `filename`
- `chunk_id`
- `text_preview`
- `score`

This keeps the API honest: answers are connected to the retrieved chunks that produced them.

## Storage

Local files are stored under `DATA_DIR`:

```text
data/
  documents.json
  uploads/
    <document_id>/
      <original_filename>
  documents/
    <document_id>.json
  chroma/
    ...
```

`DELETE /documents/{document_id}` removes the document metadata, uploaded file, chunk JSON,
and ChromaDB vectors for that document.

