[![Download from TestPyPI](https://img.shields.io/badge/TestPyPI-secure--rag-blue)](https://test.pypi.org/project/secure-rag/)

# Secure RAG

> **Warning**
> Secure RAG is an **experimental research alpha** (`0.2.0a1`).
> It is intended for privacy-aware RAG experimentation and benchmarking, not production or clinical use.

**Secure RAG** is a privacy-aware Retrieval-Augmented Generation framework for local document intelligence.

Its key research contribution is **Pre-Embedding Privacy Enforcement**, where sensitive information is detected and masked **before chunking, embedding, and vector indexing**, ensuring private data never enters the retrieval pipeline in raw form.

This allows the framework to prevent sensitive data from entering the vector store in raw form. The current system demonstrates the privacy-utility tradeoff: pre-embedding masking improves privacy but can reduce retrieval quality for identity-based queries.

---

## Features

- Pre-embedding document masking
- Optional context masking before generation
- Support for `.txt` and `.pdf` inputs
- FAISS-backed semantic retrieval
- Streaming answer generation
- Lazy loading for embedding and API clients
- Packaged Python API and CLI entry point
- Research-oriented architecture for privacy-aware RAG experiments

---

## Architecture

Secure RAG follows a privacy-first retrieval pipeline:

1. Load a local document from `.txt` or `.pdf`
2. Parse and validate input
3. Apply privacy masking before chunking
4. Split the document into overlapping chunks
5. Convert chunks into embeddings
6. Index embeddings in FAISS
7. Retrieve relevant chunks using the raw user query
8. Optionally mask retrieved context before generation
9. Stream the grounded response back to the caller

This design ensures sensitive entities can be protected **before entering the vector store**, which is the framework's core research contribution.

---

## Privacy Modes

Secure RAG supports three research modes:

- **`raw`**: no masking during indexing or generation
- **`post`**: no masking during indexing, retrieved context is masked before generation
- **`pre`**: masking is applied before chunking, embedding, and indexing

Important:
- The user query is never masked in the current design.
- `pre` mode provides the strongest privacy protection for stored content.
- `post` mode preserves raw retrieval behavior but protects context shown to the LLM.

---

## Known Limitations

- This is an experimental research framework, not a production-ready RAG package.
- Identity-based retrieval can fail in `pre` mode because raw names in queries may not align with masked indexed content.
- Some LLM backends may echo prompt structure or use outside knowledge despite grounding instructions.
- Output quality depends significantly on the configured model provider.
- Local setup may require the spaCy model `en_core_web_sm`.

---

## Installation

### From TestPyPI

```bash
pip install -i https://test.pypi.org/simple/ secure-rag
```

### Local Development

```bash
pip install -e ".[dev]"
```

### Additional Setup

Install the spaCy model:

```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

Depending on the backend you use, you may also need:
- `HF_API_KEY` / `HF_TOKEN`
- `LLM_PROVIDER=ollama` and `OLLAMA_MODEL=llama3.2`

### Docker

Secure RAG includes a ready-to-use Docker Compose setup.

**Project structure:**

```
data/
└── sample_patient_data.txt
```

The `data/` directory is mounted into the container at `/data`. Any supported document placed in this directory is accessible to the CLI.

**Quick start:**

```bash
# Build the image and start an interactive CLI session
docker compose run --rm secure-rag
```

The first run builds the image, which installs dependencies and bakes in the spaCy model. Subsequent runs are instant.

> `docker compose run --rm secure-rag` is the recommended way to interact with the CLI. It attaches directly to your terminal, handles stdin and signals correctly, and removes the container after exit.
>
> `docker compose up` is useful when running the full Compose stack (e.g. with the Ollama service via `--profile ollama`), but it is not the preferred way for interactive CLI usage.

**Changing the input document:**

Open `.env` and update `RAG_INPUT_FILE` to point to a different file inside `data/`:

```
RAG_INPUT_FILE=my_own_document.txt
```

Place the file in `data/` and run `docker compose up` again.

**Using a local Ollama instance:**

```bash
docker compose --profile ollama up
```

This starts both Secure RAG and an Ollama container. Set `LLM_PROVIDER=ollama` in `.env` to use it.

**Notes:**
- The spaCy model `en_core_web_sm` is baked into the image — no manual download needed.
- The embedding model (`all-MiniLM-L6-v2`) downloads on first use and is cached inside the container.
- Environment variables are loaded from `.env` automatically (API keys, provider, model selection).

---

## Usage

### Python API

```python
from secure_rag import build_rag, rag_answer

vector_store, chunks = build_rag("test_data.txt", use_masking=True)
answer = "".join(rag_answer("what treatment is given for chest pain?", vector_store, chunks, mask_mode="pre"))
print(answer)
```

### CLI

```bash
secure-rag test_data.txt
```

---

## Research Evaluation

Secure RAG includes a benchmark script for comparing privacy modes:

```bash
python benchmarks/privacy_eval.py
```

Current evaluation metrics:
- Document leakage
- Retrieval leakage
- Masking recall
- PHI in answers

Note: results depend on the configured model backend and local environment. LLM-based evaluation requires API or local model setup.

---

## Disclaimer

Secure RAG is a research framework for studying privacy-preserving retrieval-augmented generation. It is not intended for medical decision-making, clinical deployment, or production handling of regulated sensitive data.
