# Secure RAG — Engineering Context

## Project Status

Secure RAG is an experimental research alpha for privacy-aware retrieval-augmented generation. It is designed for experimentation, benchmarking, and release distribution through Python packaging and Docker, not for production or clinical deployment.

Current validated release state:

- Python package version: `0.2.0a1`
- TestPyPI publishing: intended release channel at this stage
- Docker runtime: implemented via `Dockerfile.runtime`
- Docker Compose workflow: implemented
- Python CI: implemented
- Docker CI: implemented
- GHCR publishing: implemented

## Project Purpose

The central research goal is to study the privacy-utility tradeoff introduced by masking sensitive entities before embedding documents.

The core design principle is privacy by design: sensitive values should be removed before they enter embeddings and the vector store. The key tradeoff is that privacy improves, but identity-based retrieval can become weaker because raw query entities no longer align with masked indexed content.

## Current Architecture

The current retrieval pipeline is:

1. Load a supported local document (`.txt` or `.pdf`)
2. Clean transcript-like artifacts from text input
3. Split the document into records
4. Optionally mask each record before chunking and embedding
5. Chunk each record independently
6. Generate embeddings with SentenceTransformers
7. Store embeddings in FAISS
8. Retrieve using the raw user query
9. Optionally mask retrieved context before generation
10. Generate a grounded answer through Hugging Face Router or Ollama
11. Return results through the Python API or interactive CLI

Supported model providers are currently:

- Hugging Face Router
- Ollama

Provider selection is environment-variable driven.

## Core RAG and Privacy Decisions

### Pre-Embedding Masking

Masking before embedding is the defining architectural choice.

Problem:

- Raw PHI in embeddings and FAISS indexes weakens storage privacy guarantees.

Decision:

- In `pre` mode, records are masked before chunking, embedding, and indexing.

Reasoning:

- This guarantees that raw sensitive values do not enter the vector store.

Tradeoff:

- Retrieval for entity-specific queries degrades because the query remains raw while indexed entities are masked.

Validation:

- This tradeoff is now treated as a core research result, not an accidental regression.

### Query Masking Is Intentionally Disabled

Problem:

- Masking queries would change the experiment and obscure the effect of masked indexed documents.

Decision:

- Queries are never masked in any mode.

Reasoning:

- The project is explicitly evaluating what happens when privacy is enforced on stored content while the user query remains natural.

Current mode definitions:

- `raw`: no masking during indexing or generation
- `post`: raw indexing, retrieved context masked before LLM generation
- `pre`: masking before chunking, embedding, and indexing

Important invariant:

- `raw` and `post` share the same raw index. `post` does not create a separate masked index.

## Chunking and Retrieval Decisions

### Record-Based Chunking

Problem:

- Whole-document chunking caused multiple patient records to be embedded and retrieved together, producing cross-record contamination.

Root cause:

- Structured patient-style input was previously treated as one flat document instead of isolated retrieval units.

Decision:

- `build_rag()` splits input into records first, then masks and chunks each record independently.

Reasoning:

- Patient records are the correct retrieval unit for this dataset shape.

Validation:

- Record-based chunking is implemented and considered essential for structured medical-style data.

### Identity-Based Retrieval Remains an Open Limitation

Problem:

- In `pre` mode, names and identifiers become placeholders during indexing, so raw identity terms in queries no longer map cleanly to stored content.

Current state:

- Condition- and treatment-oriented retrieval is stronger than identity-oriented retrieval.

Status:

- This remains an unresolved research tradeoff rather than a fixed bug.

## Masking Coverage and Fixes

### PHI Coverage

The masking flow was validated against major PHI categories before embedding in `pre` mode, including:

- person names
- phone numbers
- email addresses
- Aadhaar numbers
- PAN numbers
- DOB-style fields
- medical IDs

### Medical ID Regex Expansion

Problem:

- Compact medical ID formats leaked because earlier masking patterns were too narrow.

Validated fix:

- Medical ID masking was expanded to cover patterns such as:
  - `MRN1002`
  - `MRN 1002`
  - `MRN:1002`
  - `UHID-12345`

Reasoning:

- Structured medical records often use compact or punctuation-delimited identifiers, so masking must catch those forms before indexing or retrieval.

## Generation and Prompting Decisions

### Prompt Echo Truncation

Problem:

- Some chat-style models repeated prompt scaffolding such as `Context:`, `Question:`, or chat control tokens back into the answer.

Decision:

- `rag_answer()` post-processes generated output and truncates at known scaffold markers.

Current stop markers include:

- `\nContext:`
- `\nQuestion:`
- `[/INST]`

Tradeoff:

- This is a practical cleanup layer, not a complete guarantee against all provider-specific formatting artifacts.

### Grounding Limitations Are Model-Dependent

Current limitation:

- Even with grounding instructions, some models may still echo structure or use outside knowledge.

Status:

- The project treats this as a provider/model limitation rather than a solved infrastructure issue.

## CLI Decisions

Secure RAG provides an interactive CLI built with Typer and Rich.

### Rich Markup Must Be Disabled for Model Output

Problem:

- Some models emit bracketed tokens such as `[/ASSIST]`.

Root cause:

- Rich interprets bracketed strings as markup by default. Closing tokens from model output caused `MarkupError` during streamed printing.

Validated fix:

- Streamed model output must be printed with `markup=False`.

Reasoning:

- Model output is arbitrary text and should not be treated as terminal formatting instructions.

Tradeoff:

- Rich styling is intentionally disabled only for streamed model output. Other CLI status text can continue using Rich markup.

## Dataset and Sample Data Decisions

### Structured Sample Dataset

The project moved to larger, more realistic patient-style records with consistent fields so privacy and retrieval behavior can be tested against structured data instead of underspecified examples.

### Docker Sample Data Convention

Current Docker and Compose workflows assume the sample document lives at:

- `data/sample_patient_data.txt`

This file is mounted into containers through `./data -> /data` and is used for runtime validation and onboarding.

## Docker Runtime Decisions

### Runtime Image Design

The runtime image is built from `Dockerfile.runtime` and uses `python:3.11-slim`.

Current design decisions:

- multi-stage build
- install package from `pyproject.toml`
- non-root runtime user
- `secure-rag` as the container entrypoint
- secrets remain external

### CPU-Only PyTorch Is Mandatory in Docker

Problem:

- On Linux ARM64, default `torch` wheels pulled large NVIDIA CUDA dependency wheels even though Secure RAG uses CPU-only inference.

Root cause:

- Recent PyTorch ARM64 Linux packaging includes CUDA-related dependencies by default.

Validated fix:

- Install CPU-only `torch` first from `https://download.pytorch.org/whl/cpu`, then install the project normally so `sentence-transformers` reuses that installation.

Reasoning:

- This avoids unnecessary gigabyte-scale CUDA downloads and produces a deterministic CPU runtime.

### spaCy Model Is Baked into the Image

Decision:

- `en_core_web_sm` is installed during Docker image build.

Reasoning:

- Container users should not need a separate runtime step to install the spaCy model.

### Container User Must Have a Real Home Directory

Problem:

- Creating the runtime user with `--no-create-home` caused `PermissionError: '/home/appuser'`.

Root cause:

- Hugging Face and related libraries initialize caches under `~/.cache`. Without a real home directory, cache initialization fails.

Validated fix:

- The runtime user must be created with a writable home directory.

Reasoning:

- This is simpler and more correct than trying to special-case cache paths for each library.

## Docker Compose Decisions

### Compose Runtime Contract

Compose exists to make contributor onboarding easier without changing application code.

Current design:

- builds from `Dockerfile.runtime`
- loads environment from `.env`
- mounts `./data` to `/data`
- passes the selected document as a CLI argument
- keeps an interactive terminal experience

The `secure-rag` service runs:

- `secure-rag /data/${RAG_INPUT_FILE}`

The selected document is controlled through:

- `RAG_INPUT_FILE` in `.env` and `.env.example`

### Preferred Interactive Workflow

Decision:

- `docker compose run --rm secure-rag` is the preferred way to use the CLI.

Reasoning:

- Secure RAG is an interactive CLI, not a long-running web service. `run --rm` attaches directly to the terminal and cleans up the container afterward.

`docker compose up` remains useful when running the full Compose stack, especially when optional services such as Ollama are involved.

### Ollama Is Optional and Profile-Gated

Decision:

- The Compose `ollama` service is behind a profile and is not started by default.

Reasoning:

- Local inference support should be available without making every contributor run Ollama unnecessarily.

## CI and Release Infrastructure

### Python CI and Docker CI Are Intentionally Separate

Current CI responsibilities are intentionally split by concern.

Python CI validates:

- package installation
- pytest
- package build
- twine metadata validation

Docker CI validates:

- Docker image build health
- CLI startup via `secure-rag --help`
- mounted sample data accessibility

Reasoning:

- Package validation and container validation are different failure domains and should remain independent.

### Docker CI Is Intentionally Lightweight

Decision:

- Docker CI does not run inference, benchmarks, Hugging Face API calls, or Ollama.

Reasoning:

- Its job is infrastructure validation only.

### Docker Layer Caching Was Intentionally Deferred

Decision:

- Docker layer caching was not added to Docker CI.

Reasoning:

- For an alpha-stage project, clean and deterministic builds were preferred over additional workflow complexity and cache maintenance.

## GHCR Publishing Decisions

### Release-Driven Publishing

Decision:

- GHCR publishing triggers only on `release: published`.

Reasoning:

- Publishing container images should be an intentional release action, not a side effect of ordinary pushes.

### GHCR Image Naming Must Be Lowercase

Problem:

- GHCR rejects mixed-case repository names, but GitHub repository metadata preserves original casing.

Root cause:

- `GITHUB_REPOSITORY` may contain uppercase characters from the GitHub owner or repository name.

Validated fix:

- The GHCR workflow lowercases `GITHUB_REPOSITORY` in a shell step and reuses that normalized value for both image tags.

Current published tag strategy:

- `latest`
- release version tag (from `github.ref_name`)

Reasoning:

- This keeps tagging simple for an alpha-stage project while staying GHCR-compliant.

## Release Process

Current release flow is:

1. Update the package version
2. Commit and push the release candidate
3. Create a GitHub Release
4. Python CI validates the package
5. Docker CI validates container infrastructure
6. GHCR publish workflow pushes the official runtime image

Release policy:

- Releases are treated as immutable.
- New fixes should ship as new alpha versions rather than rewriting existing released artifacts.

## Known Limitations

- Identity-based retrieval remains weaker in `pre` mode because raw query entities do not align with masked indexed entities.
- Generation quality remains strongly dependent on the chosen model provider.
- Prompt echo is mitigated but not fully eliminated for every possible provider output format.
- Research evaluation is ongoing; the system is suitable for experimentation, not production claims.

## Current Engineering Principles

The project currently follows these operating rules:

- keep changes minimal and scoped
- use one logical commit per coherent change
- validate architectural changes with direct evidence
- prefer simple infrastructure over premature optimization
- separate research validation from infrastructure validation

## Future Work

- Improve retrieval strategies for masked identity/entity queries without weakening the privacy guarantees of `pre` mode.
- Continue empirical work around the privacy-utility tradeoff.
- Validate GHCR publishing in real release usage and keep release automation intentionally minimal.

Architectural Principle: Runtime frameworks should expose only canonical production behaviour. Experimental baselines belong to a separate evaluation harness and must never leak into the runtime API.