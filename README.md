# Secure RAG

Secure RAG is a privacy-aware retrieval-augmented generation toolkit for local document search. It combines document ingestion, chunking, embeddings, FAISS-based retrieval, masking, and streaming answer generation behind a simple Python API and CLI.

## Why Secure RAG

- Privacy-aware query and context masking
- Support for `.txt` and `.pdf` inputs
- FAISS-backed semantic retrieval
- Streaming answer generation
- Lazy loading for embedding and API clients
- Packaged Python API and CLI entry point

## Demo

Place README images in `docs/images/`.

Recommended files:
- `docs/images/demo-cli.png`
- `docs/images/demo-architecture.png`
- `docs/images/demo-streamlit.png`

### CLI Demo

![Secure RAG CLI demo](./docs/images/demo-cli.png)

The screenshot above shows the packaged terminal chat flow after indexing `data.txt`, including model loading, query entry, and streamed answer output.



## Architecture

Secure RAG follows a simple pipeline:

1. Load a local document from `.txt` or `.pdf`
2. Split the document into overlapping chunks
3. Convert chunks into embeddings
4. Index embeddings in FAISS
5. Mask the user query before retrieval
6. Retrieve relevant chunks
7. Mask retrieved context before generation
8. Stream the grounded response back to the caller

## Installation

From PyPI:

```bash
pip install secure-rag
```

For local development:

```bash
pip install -e .
```

## Quick Start

### CLI

```bash
secure-rag data.txt
```

### Python

```python
from secure_rag import build_rag, rag_answer

vector_store, chunks = build_rag("data.txt")
response = rag_answer("What is RAG?", vector_store, chunks)

print("".join(response))
```

## Environment Configuration

Answer generation requires a Hugging Face Router API key only when generation is used.

```bash
export HF_API_KEY=your_api_key
```

Optional environment variables:

- `HF_API_KEY`: API key for generation
- `HF_BASE_URL`: override the default router endpoint
- `HF_MODEL`: override the default generation model

Importing `secure_rag` does not require environment variables by itself.

## Supported Inputs

- Plain text files: `.txt`
- PDF documents: `.pdf`

## Project Structure

```text
secure_rag/
  cli.py
  embedding.py
  generator.py
  masker.py
  pdf_loader.py
  rag_pipeline.py
  retriever.py
  vector_store.py
docs/
  images/
```

## Current Capabilities

- Build a local retrieval pipeline from a document
- Query the indexed content through the CLI or Python API
- Stream generated responses token by token
- Apply basic regex-based masking for emails and phone numbers

## Known Limitations

- Masking is currently regex-based and not medical-grade de-identification
- Persistent vector storage is not implemented yet
- Source citations are not surfaced in responses yet
- Automated evaluation and medical-readiness controls are still in progress

## Development

Core package metadata is defined in [pyproject.toml](pyproject.toml). Current implementation status is tracked in [rag_features_checklist.md](rag_features_checklist.md).

## License

MIT
