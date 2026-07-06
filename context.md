# Secure RAG — Engineering Context

## Repository Philosophy

Secure RAG is a privacy-aware retrieval-augmented generation framework designed for research experimentation, not production deployment. The repository is organized around three distinct systems:

1. **Production Runtime** (`secure_rag/`) — The canonical Secure RAG pipeline shipped as a Python package
2. **Research Evaluation Framework** (`benchmarks/`) — An independent evaluation harness for comparing privacy strategies
3. **Deployment Infrastructure** (Docker, Compose, CI, GHCR) — Containerization and automation for distribution

The central research goal is to study the privacy-utility tradeoff introduced by masking sensitive entities before embedding documents. The core design principle is privacy by design: sensitive values should be removed before they enter embeddings and the vector store.

**Architectural Principle:** Runtime frameworks should expose only canonical production behaviour. Experimental baselines belong to a separate evaluation harness and must never leak into the runtime API.

---

## Repository Architecture

### Production Runtime (`secure_rag/`)

The runtime is a Python package that implements the canonical Secure RAG pipeline. It is distributed via TestPyPI and Docker. The runtime has no knowledge of benchmark configurations or research baselines.

**Public API:**
- `build_rag(file_path)` — Build a masked vector index from a document
- `rag_answer(query, vector_store, chunks)` — Generate grounded answers using the index

**Key Modules:**
- `rag_pipeline.py` — Core pipeline orchestration
- `masker.py` — PII detection and masking
- `embedding.py` — SentenceTransformer embeddings
- `vector_store.py` — FAISS indexing
- `retriever.py` — Semantic retrieval
- `generator.py` — LLM-based answer generation
- `pdf_loader.py` — Document loading and chunking
- `cli.py` — Interactive CLI entrypoint

### Research Evaluation Framework (`benchmarks/`)

The benchmark is an independent evaluation harness that compares three privacy strategies using a synthetic medical dataset. It consumes the runtime as a black box but does not define it.

**Evaluation Configurations:**
- **Baseline A** — Raw Retrieval-Augmented Generation (no masking)
- **Baseline B** — Post-Retrieval Privacy Masking (masking after retrieval)
- **Secure RAG** — Pre-Embedding Privacy Enforcement (canonical runtime)

**Key Components:**
- `privacy_eval.py` — Evaluation orchestrator with configuration registry
- `generate_dataset.py` — Synthetic Indian medical dataset generator
- `dataset.jsonl` — 120 synthetic medical records with known PII
- `dataset_queries.json` — 600+ benchmark queries
- `train_test_split.json` — Fixed train/test split for reproducibility
- `results.json` — Structured evaluation output

**Metrics:**
- Document Leakage — PII present in indexed chunks
- Retrieval Leakage (k=5) — PII in top-5 retrieved chunks
- Masking Recall — PII successfully removed by masker
- PHI in Answers — PII in LLM-generated responses

### Deployment Infrastructure

**Docker:**
- `Dockerfile.runtime` — Multi-stage build for production runtime
- CPU-only PyTorch for deterministic ARM64 builds
- spaCy model baked into image
- Non-root runtime user with writable home directory

**Docker Compose:**
- `docker-compose.yml` — Contributor onboarding workflow
- `./data` mounted to `/data` for document access
- Ollama service behind optional profile
- Preferred workflow: `docker compose run --rm secure-rag`

**CI:**
- `python-ci.yml` — Package installation, pytest, build, twine validation
- `docker-ci.yml` — Docker build health, CLI entrypoint verification
- Separate workflows for package and container validation

**GHCR:**
- `publish-ghcr.yml` — Release-driven container publishing
- Triggers only on `release: published`
- Lowercase repository naming for GHCR compliance
- Tags: `latest` and version-specific

---

## Runtime Design Principles

### Single Canonical Pipeline

The runtime exposes exactly one retrieval pipeline. There are no optional branches, no configuration modes, and no research abstractions in the public API.

**Pipeline:**
1. Load document (`.txt` or `.pdf`)
2. Clean transcript artifacts
3. Split into records
4. **Mask each record (mandatory)**
5. Chunk each record independently
6. Generate embeddings
7. Index in FAISS
8. Retrieve using raw query
9. Generate grounded answer
10. Stream response

### Mandatory Pre-Embedding Masking

Masking before embedding is the defining architectural choice. Records are always masked before chunking, embedding, and indexing. This guarantees that raw sensitive values never enter the vector store.

**Tradeoff:** Retrieval for entity-specific queries degrades because the query remains raw while indexed entities are masked. This is treated as a core research result, not an accidental regression.

### No Runtime Modes

The runtime has no concept of "privacy modes." The benchmark framework compares three evaluation configurations, but these are research baselines implemented externally, not runtime features.

### Simplified Public API

The public API is minimal:
- `build_rag(file_path)` — Single parameter, always masks
- `rag_answer(query, vector_store, chunks)` — No mode parameters

This surface area prevents benchmark concepts from leaking into the runtime.

### Query Masking Is Intentionally Disabled

Queries are never masked. The project explicitly evaluates what happens when privacy is enforced on stored content while the user query remains natural.

---

## Benchmark Design Principles

### Evaluation Framework

The benchmark is an independent evaluation harness that answers the research question: How does Secure RAG compare against alternative privacy strategies while preserving retrieval utility?

### Baseline A — Raw Retrieval-Augmented Generation

No masking is applied during indexing, retrieval, or answer generation. This measures the baseline privacy leakage of standard RAG.

**Implementation:** Composes runtime primitives directly: raw text → `chunk_text()` → `embed_chunks()` → `VectorStore` → retrieve → generate_answer

### Baseline B — Post-Retrieval Privacy Masking

Documents are indexed without masking. Retrieved context is masked immediately before answer generation. This isolates the privacy benefit of masking at inference time only.

**Implementation:** Composes runtime primitives: raw text → `chunk_text()` → `embed_chunks()` → `VectorStore` → retrieve → `mask_text()` → generate_answer

### Secure RAG — Pre-Embedding Privacy Enforcement

Sensitive entities are masked before chunking and embedding. The vector store never contains raw sensitive information. No answer-time masking is performed. This represents the canonical Secure RAG pipeline.

**Implementation:** Uses canonical runtime: `build_rag(file_path)` → `rag_answer(query, vector_store, chunks)`

### Runtime Boundary

The benchmark consumes the runtime as a black box. The dependency is one-way: `benchmarks/` imports from `secure_rag/`, but `secure_rag/` never imports from `benchmarks/`.

### Configuration Registry

Evaluation configurations are defined once in `EVALUATION_CONFIGS`. Each configuration owns:
- Stable machine identifier (`id`)
- Human-readable display name (`display_name`)
- Methodology description (`description`)
- Index/chunk source (`get_idx`)
- Answer generation logic (`answer`)

Adding a new baseline requires one entry in the registry. No scattered loops or hardcoded mode strings.

### Research Terminology

The benchmark uses approved research terminology:
- "Baseline A" instead of "raw mode"
- "Baseline B" instead of "post mode"
- "Secure RAG" instead of "pre mode"

Result JSON keys use stable identifiers: `baseline_a`, `baseline_b`, `secure_rag`.

---

## Deployment Philosophy

### Docker

Docker provides a self-contained runtime environment for contributors and users. The image is built with multi-stage optimization, CPU-only PyTorch, and the spaCy model pre-installed.

**Design decisions:**
- Multi-stage build to minimize image size
- CPU-only PyTorch to avoid unnecessary CUDA dependencies on ARM64
- spaCy model baked into image for immediate usability
- Non-root runtime user with writable home directory for cache initialization

### Docker Compose

Compose exists to make contributor onboarding easier without changing application code. It provides an interactive CLI workflow with optional local inference support.

**Preferred workflow:** `docker compose run --rm secure-rag` — attaches directly to terminal, cleans up container after exit.

### GHCR

Container publishing is release-driven, not push-driven. Images are published only when a GitHub Release is created, ensuring that published artifacts are intentional and immutable.

### CI

CI responsibilities are intentionally split:
- Python CI validates package installation, tests, build, and metadata
- Docker CI validates container build health and CLI entrypoint

This separation ensures that package and container failures are isolated and independently debuggable.

---

## Architectural Decisions

### Runtime No Longer Owns Benchmark Concepts

**Decision:** Removed `use_masking` and `mask_mode` parameters from the runtime API.

**Reasoning:** Optional masking and mode selection are benchmark concerns, not runtime concerns. The runtime should expose only the canonical Secure RAG pipeline.

**Impact:** `build_rag(file_path)` now always masks. `rag_answer(query, vector_store, chunks)` never masks at answer time.

### Benchmark Consumes Runtime

**Decision:** Benchmark implements raw and post-retrieval baselines by composing runtime primitives directly, rather than calling the runtime with mode parameters.

**Reasoning:** This preserves the runtime as a black box while allowing the benchmark to construct alternative baselines for comparison.

**Impact:** Baseline A and Baseline B use `chunk_text()`, `embed_chunks()`, `VectorStore`, and `generate_answer()` directly. Secure RAG uses the canonical `build_rag()` and `rag_answer()`.

### Identity Separated From Presentation

**Decision:** Each benchmark configuration has a stable machine identifier (`id`) separate from human-readable display names (`display_name`).

**Reasoning:** Result JSON keys should be stable for programmatic consumption, while console output should be readable for humans. Identity and presentation can evolve independently.

**Impact:** Result JSON uses `baseline_a`, `baseline_b`, `secure_rag`. Console output displays "Baseline A", "Baseline B", "Secure RAG".

### Configuration Registry

**Decision:** Centralized evaluation configuration selection into `EVALUATION_CONFIGS` registry.

**Reasoning:** Scattered hardcoded loops over mode strings made adding new baselines difficult. A registry makes configuration explicit and extensible.

**Impact:** Adding a new baseline requires one entry in `EVALUATION_CONFIGS`. All evaluation loops iterate the registry.

### Research Terminology

**Decision:** Use approved research terminology (Baseline A, Baseline B, Secure RAG) instead of implementation mode names (raw, post, pre).

**Reasoning:** Mode names are implementation details. Research terminology describes the evaluated strategies and is more meaningful to readers.

**Impact:** Documentation, console output, and result descriptions use research terminology. Internal identifiers remain stable.

### Record-Based Chunking

**Decision:** Split input into records first, then mask and chunk each record independently.

**Reasoning:** Whole-document chunking caused cross-record contamination. Patient records are the correct retrieval unit for structured medical-style data.

**Impact:** Each patient record is masked and chunked independently, preventing PII from one record leaking into chunks from another.

### CPU-Only PyTorch in Docker

**Decision:** Install CPU-only PyTorch from dedicated CPU index before installing project dependencies.

**Reasoning:** Default PyTorch ARM64 wheels include CUDA dependencies (~1GB) even for CPU-only inference. This wastes bandwidth and complicates builds.

**Impact:** Docker builds are faster, smaller, and deterministic on ARM64 platforms.

---

## Lessons Learned

### Nested Generators Can Be Unnecessary Indirection

The original `rag_answer()` used a nested `cleaned_response()` generator that buffered the full response before yielding. Flattening to a direct `yield` achieved the same behaviour with less code and clearer intent.

### Removing Optional Parameters in Steps Is Correct

Removing `use_masking` and `mask_mode` in separate steps was the right approach. Each removal affected different parts of the pipeline and had different downstream impacts. Combining them would have made validation harder.

### Tests Using Defaults Validate the Canonical Path

All tests used default argument values during the runtime refactor. This validated that the default configuration was already the intended Secure RAG behaviour. No test modifications were required.

### Benchmark Built Its Own Index

The benchmark composed runtime primitives directly rather than calling `build_rag()`. This meant the `use_masking` removal had zero impact on benchmark code initially. The only cross-boundary call was `rag_answer(mask_mode=...)`, which required Phase 3 attention.

### Docker Daemon Availability Is Not Guaranteed

Local Docker validation was deferred to CI because the daemon was unavailable on the development machine. CI covers Docker builds on push, so this was acceptable. Consider adding `make docker-test` for local validation when the daemon is available.

### Configuration Centralization Simplifies Extension

Before the configuration registry, adding a new baseline required editing multiple hardcoded loops. After centralization, adding a baseline requires one entry in `EVALUATION_CONFIGS`. This significantly reduces maintenance burden.

### Identity-Presentation Separation Enables Evolution

Separating stable identifiers from display names allows result JSON keys to remain stable while console output can be improved for readability. This is a pattern worth reusing in other contexts.

---

## Future Guidelines

### Do Not Add Benchmark Concepts to Runtime

Future contributors must not add benchmark abstractions to the runtime. The runtime should remain a black box with a single canonical pipeline. All research configurations belong in `benchmarks/`.

### Preserve One-Way Dependency

The dependency direction must remain: `benchmarks/` → `secure_rag/`. Never import from `benchmarks/` in runtime code. This preserves the runtime as a stable, independent package.

### Keep Runtime Deterministic

The runtime should have no optional branches or configuration modes. All behaviour should be deterministic and explicit. If a new behaviour is needed, it should be implemented as a separate baseline in the benchmark, not a runtime mode.

### Add New Baselines via Registry

When adding new evaluation configurations to the benchmark, add one entry to `EVALUATION_CONFIGS`. Do not add new hardcoded loops or mode strings. The registry is the single source of truth for configuration.

### Maintain Stable Result Identifiers

When adding new baselines, choose stable machine identifiers for result JSON keys. Do not change existing identifiers. This preserves comparability with historical results.

### Separate Identity from Presentation

When adding new configurations, provide both a stable identifier and a human-readable display name. Use the identifier for programmatic access and the display name for console output.

### Validate Architecture with Evidence

Before making architectural changes, validate with direct evidence. Run tests, benchmarks, and infrastructure checks. Do not rely on assumptions about behaviour.

### Prefer Minimal Changes

Make the smallest change that achieves the goal. Avoid refactoring unrelated code or adding unnecessary features. The project follows a disciplined approach to changes.

---

## Repository Evolution

The repository evolved from a monolithic codebase with benchmark concepts leaking into the runtime to a clean separation of concerns across three distinct systems.

**Initial State:**
- Runtime exposed optional masking via `use_masking` parameter
- Runtime exposed query modes via `mask_mode` parameter
- Benchmark terminology (raw, post, pre) appeared in runtime code
- Benchmark called runtime with mode parameters
- Documentation described runtime modes as features

**Refactor Outcomes:**

**Runtime:**
- Removed `use_masking` — masking is now mandatory
- Removed `mask_mode` — no answer-time masking in runtime
- Simplified API to `build_rag(file_path)` and `rag_answer(query, vector_store, chunks)`
- No benchmark terminology remains in runtime code
- Runtime is now a black box with single canonical pipeline

**Benchmark:**
- Removed runtime coupling — no longer calls `rag_answer(mask_mode=...)`
- Implements raw and post baselines by composing primitives directly
- Centralized configuration selection in `EVALUATION_CONFIGS` registry
- Separated identity from presentation (stable IDs vs display names)
- Uses approved research terminology (Baseline A, Baseline B, Secure RAG)
- Results use stable identifiers: `baseline_a`, `baseline_b`, `secure_rag`

**Documentation:**
- README updated to describe canonical pipeline
- Removed "Privacy Modes" section
- Updated Python API examples to use canonical signatures
- Research evaluation section describes benchmark configurations
- context.md updated to reflect mandatory masking and no runtime modes

**Architecture:**
- Clear separation: Runtime, Benchmark, Deployment
- One-way dependency: Benchmark → Runtime
- Runtime owns canonical pipeline
- Benchmark owns evaluation configurations
- Deployment owns distribution infrastructure

The repository now accurately reflects the approved architecture: a production runtime with mandatory pre-embedding masking, an independent research evaluation framework, and deployment infrastructure for distribution.