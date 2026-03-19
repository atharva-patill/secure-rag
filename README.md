# secure-rag

`secure-rag` is a privacy-aware retrieval-augmented generation framework for local document search with masking and streaming answer generation.

## Features

- Privacy-aware query and context masking
- Support for `.txt` and `.pdf` documents
- FAISS-backed vector search
- Streaming answer generation
- Import-safe lazy loading for embedding models and API clients

## Installation

```bash
pip install secure-rag
```

## Usage

CLI:

```bash
secure-rag data.txt
```

Python:

```python
from secure_rag import build_rag, rag_answer

vector_store, chunks = build_rag("data.txt")
response = rag_answer("What is RAG?", vector_store, chunks)

print("".join(response))
```

## Environment

Answer generation requires `HF_API_KEY` to be set only when generation is used. Importing `secure_rag` does not require network access or environment variables.
