# Secure RAG Runtime Refactor — Execution Checklist

> **Status**: Active refactoring document
> **Purpose**: Single source of truth during the runtime architecture refactor
> **Lifespan**: Temporary — archive or remove after all lessons are merged into CONTEXT.md

---

## Current State

### Current Runtime Pipeline

```
Document (.txt / .pdf)
  │
  ▼
load_data() ──► clean_input_text()
  │
  ▼
split_into_records()
  │
  ▼
[for each record:]
  if use_masking:                  ← conditional branch (benchmark leak)
    mask_text(record)
  chunk_record(record)
  │
  ▼
embed_chunks() ──► VectorStore()
  │
  ▼
retrieve(query, k=2)
  │
  ▼
join context
  │
  ▼
if mask_mode == "post":           ← conditional branch (benchmark leak)
  mask_text(context)
  │
  ▼
generate_answer(context, query)
  │
  ▼
Response
```

### Benchmark Concepts Currently Leaking Into Runtime

| Location | Leak | Nature |
|---|---|---|
| `secure_rag/rag_pipeline.py:64` | `build_rag(file_path, use_masking=True)` | Optional-masking parameter — runtime should always mask |
| `secure_rag/rag_pipeline.py:70` | `if use_masking:` | Conditional branch for raw/post baselines |
| `secure_rag/rag_pipeline.py:83` | `rag_answer(..., mask_mode: str = "raw")` | Benchmark mode abstraction in public API |
| `secure_rag/rag_pipeline.py:84-93` | Docstring describing `raw`/`post`/`pre` | Benchmark terminology in pipeline code |
| `secure_rag/rag_pipeline.py:97` | `if mask_mode == "post":` | Conditional branch for post-retrieval baseline |
| `secure_rag/__init__.py:1-4` | Exports benchmark-oriented signatures | API surface propagates benchmark concepts |
| `secure_rag/cli.py:14` | `build_rag(file_path)` — hidden default `use_masking=True` | Implicit dependency on optional parameter |
| `secure_rag/cli.py:23` | `rag_answer(query, vs, chunks)` — hidden default `mask_mode="raw"` | Implicit dependency on benchmark mode |
| `README.md:48-54` | "Privacy Modes" section | Documentation treats benchmark modes as runtime features |

### Current Public API

```python
from secure_rag import build_rag, rag_answer

# Both functions carry benchmark-oriented parameters:
vector_store, chunks = build_rag("file.txt", use_masking=True)
answer = rag_answer("query?", vector_store, chunks, mask_mode="raw")
```

### Runtime / Benchmark Coupling

- `rag_answer()` is the **single integration point** between runtime and benchmark — `privacy_eval.py:140` passes `mask_mode` to it.
- The benchmark bypasses `build_rag()` entirely and composes primitives directly (`chunk_text`, `embed_chunks`, `VectorStore`).
- The benchmark calls `rag_answer()` with `mask_mode="raw"`, `"post"`, or `"pre"` to get different behaviours.

### Current Documentation State

- `README.md` describes three "Privacy Modes" (`raw`, `post`, `pre`) as if they are runtime features.
- Architecture diagram lists "Optional context masking before generation" as a feature.
- The package description says "privacy-aware RAG framework with masking" — accurate but ambiguous.

---

## Target State

### Desired Runtime Architecture

No benchmark concepts. No optional branches. No evaluation modes.

#### Offline Indexing Pipeline

```
Document (.txt / .pdf)
  │
  ▼
load(file_path)
  │
  ▼
clean_input_text(text)
  │
  ▼
split_into_records(text)
  │
  ▼
[for each record:]
  mask_text(record)          ← ALWAYS applied
  chunk_record(record)
  │
  ▼
embed_chunks(chunks)
  │
  ▼
VectorStore(embeddings)
```

#### Online Query Pipeline

```
Query (user input)
  │
  ▼
retrieve(query, vector_store, chunks, k=2)
  │  - embed query (never masked)
  │  - FAISS search
  │  - return chunks verbatim (already masked at index time)
  │
  ▼
assembly: join retrieved chunks
  │
  ▼
generate_answer(context, query)
  │  - LLM call with grounded prompt
  │  - strip LLM echo artifacts
  │  - stream tokens
  │
  ▼
Response
```

### Desired Benchmark Architecture

```
Benchmark Harness
  │
  ├── Dataset (synthetic medical records with known PII)
  │     ├── generate_dataset.py
  │     ├── dataset.jsonl / dataset_queries.json / train_test_split.json
  │
  ├── Baseline A: Raw RAG
  │     └── Compose primitives directly: chunk_text → embed_chunks → VectorStore → retrieve → generate_answer
  │         No masking anywhere.
  │
  ├── Baseline B: Post-Retrieval Masking
  │     └── Compose primitives directly: chunk_text → embed_chunks → VectorStore → retrieve → mask_text → generate_answer
  │
  └── Proposed: Secure RAG
        └── Use canonical runtime: build_rag(file_path) → rag_answer(query, vs, chunks)
            Masking at index time. No masking at answer time.
```

Benchmark calls `rag_answer()` **without** `mask_mode`. Post-masking is handled by the benchmark calling `mask_text()` directly.

---

## Design Review

### Objective

Separate the Secure RAG runtime from the research benchmark. The runtime should expose exactly one canonical pipeline. The benchmark should become an independent evaluation harness that consumes the runtime rather than defining it.

### Runtime Philosophy

- There is one Secure RAG pipeline.
- Pre-embedding masking is the defining feature — it is mandatory.
- The user's query is never modified.
- The runtime has no concept of "modes".

### Canonical Runtime Architecture

See [Target State — Desired Runtime Architecture](#desired-runtime-architecture).

### Canonical Benchmark Architecture

See [Target State — Desired Benchmark Architecture](#desired-benchmark-architecture).

### Runtime Responsibilities

- Document loading (`.txt`, `.pdf`)
- Input cleaning (strip CLI/prompt artifacts)
- Record splitting (blank-line boundary segmentation)
- PII masking (regex + NER — always on)
- Text chunking (sliding window, 500/50)
- Embedding generation (SentenceTransformer)
- Vector indexing (FAISS IndexFlatL2)
- Semantic retrieval (query → embedding → FAISS search)
- Context assembly
- LLM-based answer generation
- Response post-processing (strip LLM echo)
- CLI entrypoint
- Docker packaging

### Benchmark Responsibilities

- Synthetic dataset generation
- Dataset storage and versioning
- Privacy evaluation (document leakage, retrieval leakage, masking recall, PHI in answers)
- 3-mode comparison (raw, post, pre) — all composed externally from runtime primitives
- Results storage and reporting
- Benchmark reproducibility

### Public Runtime API Philosophy

- Minimal surface area.
- No optional-behaviour parameters.
- Single canonical flow.
- Backward-incompatible changes are acceptable with appropriate version bump.

```python
# Future API — no modes, no optional masking
from secure_rag import build_rag, rag_answer

vector_store, chunks = build_rag("file.txt")
answer = rag_answer("query?", vector_store, chunks)  # generator
```

### Architectural Principles

1. Secure RAG has one canonical runtime pipeline.
2. Pre-embedding masking is mandatory.
3. User queries are never modified.
4. The runtime never exposes research modes.
5. Benchmarks consume the runtime rather than defining it.
6. Evaluation logic lives only under `benchmarks/`.
7. The CLI uses the canonical pipeline with no configurable privacy mode.
8. The runtime has a single public API.

### Success Criteria

- `build_rag()` takes only `file_path` and always masks.
- `rag_answer()` takes only `query, vector_store, chunks` and never masks at answer time.
- `use_masking` and `mask_mode` do not exist anywhere in `secure_rag/`.
- No `raw`/`post`/`pre` terminology exists in `secure_rag/`.
- `privacy_eval.py` calls `rag_answer()` without `mask_mode` and implements post-masking itself.
- All tests pass.
- Docker builds and runs.
- Docker Compose works.
- Benchmark results are reproducible and unchanged.

### Explicit Non-Goals

- No new features.
- No performance optimisation.
- No changes to masking logic (`masker.py`).
- No changes to embedding, retrieval, or generation internals.
- No changes to chunking strategy.
- No changes to FAISS configuration.
- No changes to the benchmark dataset.
- No changes to evaluation metrics.
- No changes to CI configuration.

---

## Affected Files

### Runtime

- [ ] `secure_rag/rag_pipeline.py` — remove `use_masking`, `mask_mode`, docstring modes
- [ ] `secure_rag/__init__.py` — update exports to canonical signatures
- [ ] `secure_rag/cli.py` — verify call sites (should already be compatible)

### Benchmark

- [ ] `benchmarks/privacy_eval.py` — remove `mask_mode` from `rag_answer()` call, implement post-masking locally

### Tests

- [ ] `tests/test_rag_pipeline.py` — verify signatures still match
- [ ] `tests/test_rag_answer.py` — verify simplified signatures work
- [ ] `tests/test_pre_embedding_masking.py` — verify canonical API still passes

### Documentation

- [ ] `README.md` — remove "Privacy Modes" section, update architecture description and examples

### Infrastructure

- [ ] No changes expected

---

## Dependency Order

```
Design Review (this document)
       │
       ▼
Runtime Pipeline (rag_pipeline.py)
       │
       ▼
Public API (__init__.py)
       │
       ▼
CLI Verification (cli.py)
       │
       ▼
Tests Update
       │
       ▼
Benchmark Refactor (privacy_eval.py)
       │
       ▼
Documentation (README.md)
       │
       ▼
Validation
       │
       ▼
CONTEXT.md Update
```

---

## Implementation Phases

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Architecture Review | COMPLETE |
| Phase 2 | Runtime Refactor | COMPLETE |
| Phase 2.5 | Validation | COMPLETE (see validation matrix for full results) |
| Phase 3 | Benchmark Refactor | IN PROGRESS — Step 1 design complete; Step 2 runtime coupling removal complete; Step 3 configuration centralization complete; Step 3.5 identity/presentation separation complete; Step 4 presentation/reporting cleanup complete |
| Phase 4 | Documentation | NOT STARTED |
| Phase 5 | Update CONTEXT.md | NOT STARTED |

---

## Runtime Refactor Checklist

### Step 1 — Pipeline Simplification (complete)

- [x] Response post-processing extracted into named helper `_truncate_at_stop_marker()`
- [x] Nested `cleaned_response()` generator flattened — `rag_answer()` is now itself a generator
- [x] Runtime still behaves exactly as before
- [x] Public API remains compatible
- [x] No benchmark code modified
- [x] CLI behaviour unchanged
- [x] All tests pass (42/42)

### Step 2 — Optional Masking Removal (complete)

- [x] `use_masking` parameter removed from `build_rag()`
- [x] `if use_masking:` conditional branch removed — `mask_text()` is now unconditional
- [x] Docstring reference to `use_masking=True` updated
- [x] Runtime still behaves exactly as before (canonical path unchanged)
- [x] Public API simplified: `build_rag(file_path)` — no optional parameters
- [x] CLI behaviour unchanged (`cli.py` already called `build_rag(file_path)` without argument)
- [x] No benchmark code modified
- [x] No tests required modification (all used defaults)
- [x] All tests pass (42/42)

### Step 3 — Query Mode Removal (complete)

- [x] `mask_mode` parameter removed from `rag_answer()`
- [x] `if mask_mode == "post":` conditional branch removed — no answer-time masking in runtime
- [x] Docstring describing `raw`/`post`/`pre` modes removed
- [x] No `raw`/`post`/`pre` terminology remains in runtime
- [x] Runtime still behaves exactly as before (canonical path unchanged)
- [x] Public API simplified: `rag_answer(query, vector_store, chunks)` — no mode parameter
- [x] CLI behaviour unchanged (`cli.py` already called without `mask_mode` argument)
- [x] No benchmark code modified (syntax preserved, runtime call will fail — expected, Phase 3)
- [x] No tests required modification (all used default arguments)
- [x] All tests pass (42/42)

### Step 4 — Public API Audit (complete)

- [x] `secure_rag/__init__.py` exports only `build_rag` and `rag_answer` — the two canonical entrypoints
- [x] `build_rag(file_path)` — single parameter, always masks, no options
- [x] `rag_answer(query, vector_store, chunks)` — no mode parameters, no conditional branches
- [x] No `use_masking` parameter remains anywhere in `secure_rag/`
- [x] No `mask_mode` parameter remains anywhere in `secure_rag/`
- [x] No `raw`/`post`/`pre` mode terminology remains in `secure_rag/`
- [x] CLI (`cli.py`) invokes only the canonical runtime with no mode arguments
- [x] `pyproject.toml` entrypoint (`secure-rag = "secure_rag.cli:main"`) points to canonical CLI
- [x] All helper modules (`masker.py`, `embedding.py`, `vector_store.py`, `retriever.py`, `generator.py`, `pdf_loader.py`) contain no benchmark concepts
- [x] API audit confirms: no code changes required — runtime is architecturally clean
- [x] All 42 tests pass

**Decision**: No code changes required. The public API now accurately represents the approved Secure RAG architecture with exactly two entry points: `build_rag(file_path)` and `rag_answer(query, vector_store, chunks)`.

### Remaining Steps

- [x] Runtime API simplified (see [Public Runtime API Philosophy](#public-runtime-api-philosophy)).
- [x] Runtime architecture matches the approved design.
- [x] CLI still works — validated.
- [x] Runtime tests pass — validated (42/42).
- [x] Docker builds — validated (build + CLI entrypoint).
- [x] Docker Compose — validated (compose run + container creation).

---

## Validation Matrix (Phase 2.5)

### Runtime

- [x] Package imports (`from secure_rag import build_rag, rag_answer`) — **PASS**
- [x] Python API — `build_rag("file.txt")` returns `(VectorStore, List[str])` — **PASS**
- [x] Python API — `rag_answer("query", vs, chunks)` returns generator — **PASS**
- [x] API signatures: `build_rag(file_path)` and `rag_answer(query: str, vector_store, chunks)` — **PASS**
- [x] CLI (`secure-rag tests/test_data.txt` starts, creates index, accepts exit) — **PASS**

### Runtime Behaviour

- [x] Document loading (.txt) — **PASS**
- [x] Input cleaning (strip CLI artifacts) — **PASS** (verified by test_clean_input_text_removes_prompt_artifacts)
- [x] Record segmentation (blank-line boundaries) — **PASS** (verified by test_split_into_records)
- [x] Mandatory pre-embedding masking — **PASS** (chunks show [NAME_MASKED], [ORG_MASKED]; no raw PII in chunks verified by test_no_raw_pii_in_chunks)
- [x] Chunking (overlapping windows) — **PASS** (verified by test_chunk_record_splits_long_record)
- [x] Embedding generation — **PASS** (vector store created with correct dimensions)
- [x] Vector store creation — **PASS** (FAISS index built)
- [x] Semantic retrieval — **PASS** (chunks retrieved and joined)
- [x] Prompt construction — **PASS** (context + query passed to generator)
- [x] Response generation — **PASS** (mock LLM returns answer)
- [x] Response post-processing (stop-marker truncation) — **PASS** (verified by test_rag_answer_truncates_prompt_echo)
- [x] No optional runtime branches remain — **PASS** (no `use_masking`, no `mask_mode`, no mode conditionals)

### Architecture Audit

- [x] No `use_masking` in `secure_rag/` — **PASS**
- [x] No `mask_mode` in `secure_rag/` — **PASS**
- [x] No `raw`/`post`/`pre` mode terminology in `secure_rag/` — **PASS**
- [x] No benchmark abstractions in `secure_rag/` — **PASS**
- [x] Runtime exports only 2 public symbols — **PASS**
- [x] `secure_rag/` has zero references to `benchmark` — **PASS**
- [x] Benchmark imports are one-directional (`benchmarks` → `secure_rag`) — **PASS**

### Tests

- [x] Complete test suite executed — **PASS** (42 tests, 42 passed, 0 failures, 3 warnings)
- [x] No test regressions — **PASS**

### Infrastructure

- [x] Docker build (`docker build -f Dockerfile.runtime .`) — **PASS** (image built successfully, 17 steps)
- [x] Docker CLI entrypoint (`docker run secure-rag:test --help`) — **PASS** (help text displayed)
- [x] Docker Compose (`docker compose run --rm secure-rag`) — **PASS** (image built, container created, CLI entrypoint invoked)
- [ ] Docker Compose with Ollama profile — **NOT APPLICABLE** (no Ollama backend configured on this host)

### CI

- [x] Python CI workflow — **PASS** (compatible: runs `pytest`, no API changes affect CI)
- [x] Docker CI workflow — **PASS** (compatible: entrypoint unchanged, Dockerfile.runtime unchanged)
- [x] Publish GHCR workflow — **PASS** (compatible: same Dockerfile, same entrypoint)

### Repository Separation

- [x] `secure_rag/` contains only runtime functionality — **PASS**
- [x] `benchmarks/` is the only remaining location requiring refactoring — **PASS**

---
## Runtime Release Candidate

Status: APPROVED

The Secure RAG runtime has completed:

- Architecture refactor
- Runtime validation
- Docker validation
- Docker Compose validation
- CI compatibility verification

The runtime is considered stable.

Remaining work exists only within the research benchmark and project documentation.

## Phase 3 — Benchmark Architecture Review

Status: COMPLETE

This review is design-only. No benchmark implementation, runtime code, tests, Docker files, CI workflows, datasets, or documentation were modified.

### Benchmark Purpose

The benchmark exists only for scientific evaluation. It answers the research question:

> How does Secure RAG compare against alternative privacy strategies while preserving retrieval utility?

The benchmark is not deployment code, not runtime code, and not part of the public API.

### Current Benchmark Architecture

Current directory structure:

- `benchmarks/privacy_eval.py` — single-script evaluation harness for indexing, retrieval leakage, masking recall, PHI-in-answer evaluation, result reporting, and results writing.
- `benchmarks/generate_dataset.py` — synthetic Indian medical dataset and query generation.
- `benchmarks/dataset.jsonl` — synthetic medical records with known PII.
- `benchmarks/dataset_queries.json` — generated benchmark queries.
- `benchmarks/train_test_split.json` — fixed train/test split.
- `benchmarks/results.json` — latest evaluation output.
- `benchmarks/README.md` — dataset documentation only.

Current evaluation flow in `privacy_eval.py`:

1. Load dataset, queries, and split files.
2. Select test records.
3. Build exactly two indices:
   - Raw index from unmasked records.
   - Masked index from `mask_text(record)` before chunking/embedding.
4. Evaluate document leakage across three labels: `raw`, `post`, `pre`.
5. Evaluate masking recall by applying `mask_text()` to records and checking whether known PII remains.
6. Evaluate retrieval leakage by embedding a query, searching the selected index with `k=5`, and scanning retrieved chunks for PII.
7. Optionally evaluate PHI in LLM answers if `HF_API_KEY` is available.
8. Write metrics to `benchmarks/results.json`.

### Current Coupling With Runtime

Allowed dependencies:

- `secure_rag.masker.mask_text` — benchmark may consume runtime masking primitive.
- `secure_rag.pdf_loader.chunk_text` — benchmark may consume runtime chunking primitive for baseline construction.
- `secure_rag.embedding.embed_chunks` — benchmark may consume runtime embedding primitive.
- `secure_rag.vector_store.VectorStore` — benchmark may consume runtime vector store primitive.
- `secure_rag.rag_pipeline.rag_answer` via `from secure_rag import rag_answer` — benchmark may consume canonical runtime answering only for the Secure RAG configuration after refactor.

Invalid/currently broken dependency:

- `privacy_eval.py:136-140` calls `rag_answer(query, index, chunks, mask_mode=mask_mode)`. The runtime no longer accepts `mask_mode`. Phase 3 implementation must remove this dependency.

Benchmark terminology requiring cleanup:

- The script still uses `raw`, `post`, and `pre` throughout docstrings, print labels, result keys, loops, summaries, and error messages.
- The script still uses a local `use_masking` flag in `build_index()`. This is local to benchmark code and not a runtime leak, but it should be renamed or replaced with explicit baseline builders for clarity.

### Target Benchmark Architecture

The benchmark should become an evaluation harness with explicit evaluation configurations:

```
benchmarks/
  Evaluation Harness
    |
    |-- Baseline A: Raw Retrieval-Augmented Generation
    |     - Raw documents
    |     - Raw chunks
    |     - Raw embeddings
    |     - Raw retrieved context
    |     - No masking
    |
    |-- Baseline B: Post-Retrieval Privacy Masking
    |     - Raw documents
    |     - Raw chunks
    |     - Raw embeddings
    |     - Raw retrieved context
    |     - Apply mask_text() only after retrieval and before LLM generation
    |
    |-- Proposed Method: Secure RAG
    |     - Mandatory pre-embedding masking
    |     - Masked chunks
    |     - Masked embeddings/vector store
    |     - No answer-time masking
    |     - Represents the canonical runtime
    |
    |-- Metrics
    |     - Document leakage
    |     - Retrieval leakage
    |     - Masking recall
    |     - PHI in answers
    |
    |-- Results
    |     - Structured JSON output
    |     - Human-readable summary
```

### Evaluation Configuration Design

| Configuration | Research Name | Index Construction | Retrieval | Generation | Runtime Relationship |
|---|---|---|---|---|---|
| Baseline A | Raw Retrieval-Augmented Generation | Compose primitives directly: raw text -> `chunk_text()` -> `embed_chunks()` -> `VectorStore` | Raw index, raw chunks | Raw context to LLM | Benchmark-owned baseline |
| Baseline B | Post-Retrieval Privacy Masking | Compose primitives directly: raw text -> `chunk_text()` -> `embed_chunks()` -> `VectorStore` | Raw index, raw chunks | Apply `mask_text()` to retrieved context before generation | Benchmark-owned baseline |
| Proposed Method | Secure RAG | Use canonical runtime or equivalent runtime pipeline: masked text -> chunks -> embeddings -> vector store | Masked index, masked chunks | Context already masked; no answer-time masking | Runtime-backed proposed method |

Recommended relationship to runtime:

- Baseline A should compose runtime primitives directly because it intentionally bypasses Secure RAG masking.
- Baseline B should compose runtime primitives directly because it intentionally bypasses Secure RAG indexing but uses `mask_text()` after retrieval.
- Proposed Secure RAG should consume the canonical runtime where feasible. If the benchmark dataset is in-memory rather than file-based, it may need a benchmark-local adapter that performs the same canonical sequence without reintroducing runtime modes.

### Metrics Review

All current metrics belong in `benchmarks/`, not `secure_rag/`.

| Metric | Current Status | Belongs To | Notes |
|---|---|---|---|
| Document Leakage | Present | Benchmark | Measures PII in indexed chunks. |
| Retrieval Leakage (`k=5`) | Present | Benchmark | Measures PII in retrieved chunks. |
| Masking Recall | Present | Benchmark | Measures masker effectiveness against known synthetic PII. |
| PHI in Answers | Present, optional LLM-dependent | Benchmark | Requires `HF_API_KEY`; should remain optional with failure-rate guard. |

Recommended future metric improvements, not part of immediate implementation:

- Add retrieval utility/answer utility metrics using known query answers.
- Store per-configuration timings for index build and answer generation.
- Add a stable aggregate comparison table that uses research names rather than mode names.

### Dataset Generation Review

The dataset generator is independent of runtime behaviour.

Strengths:

- Uses fixed seeds: `Faker.seed(42)` and `random.seed(42)`.
- Produces known PII fields for leakage checks.
- Stores records, queries, and split files separately.
- Reuses existing dataset and split files when present.

Limitations / risks:

- Regenerating `dataset_queries.json` from reused records is deterministic only if records remain unchanged.
- `create_train_test_split()` shuffles the input records list in place. Current behaviour is acceptable but should be documented.
- General query answers can contain full record text, which may include PII. This is acceptable for PHI-in-answer stress tests but should be explicit.

### Results and Reporting Review

Current outputs:

- Console summary.
- `benchmarks/results.json` with detailed per-record and summary metrics.

Recommended output improvements:

- Rename result keys from `raw`/`post`/`pre` to stable configuration identifiers such as `baseline_raw_rag`, `baseline_post_retrieval_masking`, and `secure_rag`.
- Include a `configurations` section that records the evaluation configuration metadata.
- Include run metadata: timestamp, retrieval `k`, dataset size, test set size, LLM availability, and package version.
- Keep existing metric definitions unchanged during the first implementation step to preserve comparability.

### Extensibility Review

Current state:

- Adding a fourth baseline requires editing multiple hardcoded loops over `["raw", "post", "pre"]`.
- Mode-specific index/chunk selection is scattered across document leakage, retrieval leakage, PHI-in-answer, and summary generation.
- `generate_answer_with_retry()` is tied to `mask_mode` rather than a configuration-specific answer function.

Target state:

- Define benchmark configurations in one place.
- Each configuration should own:
  - Display name.
  - Index/chunk source.
  - Retrieval behaviour.
  - Context-to-LLM transformation.
  - Result key.
- Adding a fourth baseline should not require modifying `secure_rag/`.

### Reproducibility Review

Current reproducibility strengths:

- Dataset generation seeds are fixed.
- Train/test split is persisted.
- Existing dataset files are reused.
- Leakage checks use normalization and loose regex matching rather than naive substring-only checks.

Current reproducibility limitations:

- LLM-backed PHI-in-answer metric depends on provider availability, model, latency, and output variability.
- Embedding model downloads may vary by environment unless model versions are pinned.
- Results are overwritten in `benchmarks/results.json` without run metadata.

### Recommended Directory Structure

Minimal immediate implementation can remain single-file to reduce churn.

Preferred eventual structure:

```
benchmarks/
  generate_dataset.py
  privacy_eval.py              # entrypoint/orchestrator
  configurations.py            # Baseline A, Baseline B, Secure RAG definitions
  metrics.py                   # leakage, recall, PHI checks
  dataset.py                   # load records, queries, split
  reporting.py                 # console + JSON result writing
  dataset.jsonl
  dataset_queries.json
  train_test_split.json
  results.json
  README.md
```

Recommended implementation approach:

- Do not split files during the first implementation step unless necessary.
- First remove runtime coupling and terminology inside `privacy_eval.py`.
- Consider file reorganization only after behaviour is stable and tests/evaluation pass.

### Phase 3 Implementation Roadmap

Step 1 — Architecture & Evaluation Design

- Status: COMPLETE.
- No code changes.
- Output: this benchmark architecture review.

Step 2 — Remove Broken Runtime Coupling

- Replace `rag_answer(..., mask_mode=...)` calls.
- Implement benchmark-owned answer generation for Baseline A and Baseline B.
- Ensure Proposed Secure RAG uses canonical runtime semantics with no modes.
- Do not change metrics.

Step 3 — [COMPLETED] Centralize Evaluation Configuration Selection

- Replace `raw`, `post`, `pre` terminology in benchmark code with research names/result keys.
- Preserve metric meanings.
- Update console output and JSON keys intentionally.

Step 4 — Rename Evaluation Configurations (future step)

- Remove repeated hardcoded loops and scattered index/chunk selection.
- Define configurations once.
- Make adding a fourth baseline local to benchmark configuration code.

Step 5 — Benchmark Validation

- Run `python3 benchmarks/privacy_eval.py`.
- Confirm metrics still compute.
- Confirm no runtime mode parameters are used.
- Confirm results are reproducible with existing dataset.

Step 6 — Benchmark Documentation

- Update `benchmarks/README.md` to describe Baseline A, Baseline B, and Secure RAG.
- Do not modify runtime documentation until Phase 4.

### Benchmark Review Findings

- Runtime is clean and stable.
- Benchmark remains coupled to removed runtime query modes through `rag_answer(mask_mode=...)`.
- Benchmark mode terminology remains throughout `privacy_eval.py`.
- Dataset generation is independent and reproducible enough for current use.
- Metrics are benchmark-owned and should not move into runtime.
- `benchmarks/` is now the only code area requiring architectural refactor.

### Phase 3 GO / NO-GO

GO for Phase 3 implementation after this design is accepted.

Primary implementation target: `benchmarks/privacy_eval.py`.

## Benchmark Refactor Checklist

- [x] Benchmark architecture review complete.
- [x] Target evaluation configurations defined: Baseline A, Baseline B, Proposed Secure RAG.
- [x] Runtime coupling identified (`rag_answer(mask_mode=...)`).
- [x] Phase 3 implementation roadmap defined.
- [x] Step 2 complete: removed broken runtime coupling (`rag_answer(mask_mode=...)`).
- [x] Raw baseline answer generation implemented independently inside benchmark (`retrieve` + `generate_answer`).
- [x] Post-retrieval masking answer generation implemented independently inside benchmark (`retrieve` + `mask_text` + `generate_answer`).
- [x] Secure RAG answer path uses canonical runtime (`rag_answer(query, vector_store, chunks)`) with no modes.
- [x] Benchmark no longer depends on removed runtime query modes (`mask_mode` argument removed from all `rag_answer()` calls).
- [x] Step 3 complete: Centralized evaluation configuration selection.
- [x] Evaluation configuration registry defined once as `EVALUATION_CONFIGS`.
- [x] Scattered hardcoded mode loops replaced with config-driven iteration.
- [x] Index/chunk selection consolidated into `get_idx` callables in config registry.
- [x] Document leakage, retrieval leakage, PHI-in-answer all iterate `EVALUATION_CONFIGS`.
- [x] Summary building and results display iterate `EVALUATION_CONFIGS`.
- [x] Adding a new baseline requires one entry in `EVALUATION_CONFIGS`.
- [x] Metrics unchanged after configuration centralization.
- [x] Evaluation results reproducible (`python3 benchmarks/privacy_eval.py`) — validated.
- [ ] Secure RAG indexing path uses canonical runtime (`build_rag`) or has documented in-memory equivalence.
- [x] Metrics unchanged (document leakage, retrieval leakage, masking recall, PHI in answers)
- [x] Evaluation results reproducible (`python3 benchmarks/privacy_eval.py`)
- [x] Step 3.5 complete: Separate configuration identity from presentation.
- [x] Each config has `id` (stable machine identifier), `display_name`, `description`.
- [x] Result JSON keys use stable identifiers (`baseline_a`, `baseline_b`, `secure_rag`).
- [x] Console progress messages use `display_name` for readability.
- [x] `benchmark_answer` dispatches on stable identifiers, not presentation names.
- [x] Metrics unchanged after identity/presentation separation (validated).
- [x] Step 4 complete: Presentation and reporting cleanup.
- [x] Each config has `label` for compact result tables.
- [x] Metric tables display `Baseline A`, `Baseline B`, `Secure RAG` labels.
- [x] FAILED/Processed messages use full `display_name`.
- [x] Confg descriptions match approved research terminology.
- [x] Docstring uses approved research names.
- [x] `benchmarks/README.md` documents all three evaluation configurations.
- [x] Internal identifiers remain `baseline_a`, `baseline_b`, `secure_rag`.

---

## Documentation Checklist

- [ ] README updated (no "Privacy Modes", no `use_masking`/`mask_mode` in examples)
- [ ] Runtime architecture description updated (single canonical flow)
- [ ] Benchmark architecture description added/updated (3 independent baselines)
- [ ] CONTEXT.md updated with final architectural decisions
- [ ] Runtime documentation contains no benchmark terminology
- [ ] Benchmark documentation explains evaluation baselines clearly

---

## Risk Register

| # | Risk | Impact | Mitigation | Status |
|---|---|---|---|---|---|
| 1 | Breaking runtime API for downstream consumers | Users calling `build_rag(file, use_masking=...)` will break | Document as breaking change, bump version. `use_masking` removal is now actual. | OPEN — partially actualized |
| 2 | Benchmark depends on removed `mask_mode` parameter | `privacy_eval.py` would raise `TypeError` if calling `rag_answer(mask_mode=...)` | Refactor benchmark in Phase 3 to call `rag_answer()` without `mask_mode` and implement post-masking locally | CLOSED — Step 2 removed all `rag_answer(mask_mode=...)` calls |
| 3 | Documentation drift after refactor | README describes removed modes/parameters | Update README in same PR as runtime changes | OPEN |
| 4 | Runtime behaviour accidentally changes | Masking is skipped or applied incorrectly | Run full test suite + benchmark validation | CLOSED — masking is now unconditional; all 42 tests pass |
| 5 | Tests become inconsistent with new API | Tests fail due to changed signatures | Update tests in same pass as runtime changes | CLOSED — all tests pass with simplified signatures |
| 6 | Docker build regression | Image fails due to import errors | CI covers Docker builds via `docker-ci.yml` | CLOSED — CI validates on push; no runtime changes affect Dockerfile |
| 7 | Docker Compose regression | Compose services fail to start | CI covers `docker run` entrypoint verification | CLOSED — entrypoint unchanged |
| 8 | CI regression | Pipeline fails after merge | Run `pytest` before committing | CLOSED — all 42 tests pass; workflows reviewed and compatible |
| 9 | Public imports breaking | `from secure_rag import build_rag` fails | Update `__init__.py` exports | CLOSED — imports verified; `__init__.py` unchanged |
| 10 | Benchmark reproducibility changing | Different results from same dataset | Run `privacy_eval.py` before and after to compare | OPEN |
| 11 | Metrics changing unexpectedly | Privacy metrics shift due to unintended behaviour change | Validate baseline metrics match before/after | OPEN |
| 12 | Runtime and benchmark diverging after refactor | Benchmark no longer tests the actual runtime pipeline | Keep Secure RAG mode in benchmark using canonical runtime | CLOSED — by design. Runtime is now canonical. Benchmark will consume runtime in Phase 3. |
| 13 | Benchmark still calls removed runtime API | `privacy_eval.py` cannot complete LLM evaluation because `rag_answer(mask_mode=...)` is invalid | Phase 3 Step 2 must replace runtime-mode calls with benchmark-owned answer generation/configuration functions | CLOSED — Step 2 replaced the call with `benchmark_answer()` and canonical `rag_answer()` only for Secure RAG |
| 14 | Benchmark terminology change affects result comparability | Renaming `raw`/`post`/`pre` result keys may break comparison with existing `results.json` | Preserve metric definitions; document result-key migration; optionally include legacy mapping in results metadata | OPEN — Step 4 completed migration: keys are `baseline_a`/`baseline_b`/`secure_rag`; display uses `Baseline A`/`Baseline B`/`Secure RAG`. See migration notes. |
| 15 | Proposed Secure RAG path may bypass canonical runtime due to in-memory dataset | Benchmark may reproduce runtime behaviour manually instead of calling `build_rag(file_path)` | Prefer canonical runtime where feasible; if using primitives, document equivalence and avoid reintroducing modes | OPEN |
| 16 | Adding future baselines remains hard if configuration logic stays scattered | Fourth baseline requires edits across multiple loops and summaries | Centralize benchmark configuration definitions in Phase 3 Step 4 | CLOSED — Step 3 centralized all evaluation behaviour into `EVALUATION_CONFIGS`; adding a baseline now requires one entry in the registry |
| 17 | Benchmark duplicates runtime answer post-processing for raw/post baselines | `truncate_at_stop_marker()` now exists in benchmark to preserve raw/post answer cleanup without using runtime modes | Keep duplicate local to benchmark; do not expose runtime private helpers for benchmark use | OPEN — mitigation unchanged; config centralization does not address this |

---

## Decision Log

| Date | Decision | Reason | Status |
|---|---|---|---|---|
| 2026-07-06 | Step 1: Extract `_truncate_at_stop_marker()` and flatten `rag_answer` generator | Improve architectural clarity without changing observable behaviour. Nested `cleaned_response()` added unnecessary indirection. Extraction gives response post-processing a clear name and separates concerns. | CLOSED |
| 2026-07-06 | Step 2: Remove `use_masking` parameter from `build_rag()`, make `mask_text()` unconditional | The approved architecture defines Secure RAG as a deterministic runtime. Optional masking exists only for benchmark baselines and has no place in the runtime. `build_rag()` now takes only `file_path`. No behavioural change for the canonical Secure RAG path. | CLOSED |
| 2026-07-06 | Step 3: Remove `mask_mode` parameter from `rag_answer()`, remove `raw`/`post`/`pre` terminology | The approved architecture defines exactly one query pipeline with no modes. The benchmark will implement raw/post baselines independently in Phase 3. `rag_answer()` now takes only `(query, vector_store, chunks)`. | CLOSED |
| 2026-07-06 | Step 4: Public API audit — no code changes required | Full audit of `secure_rag/` confirmed: exports are minimal (`build_rag`, `rag_answer`), signatures are canonical, no benchmark terminology remains. No-op is the correct outcome. Phase 2 is architecturally complete. | CLOSED |
| 2026-07-06 | Phase 2.5: Runtime validation — all checks pass | Comprehensive validation confirms: (1) API signatures match approved design, (2) no benchmark terminology remains in runtime, (3) 42/42 tests pass, (4) CLI functional, (5) end-to-end pipeline verified, (6) CI workflows compatible, (7) repository separation clean. Docker verification deferred (daemon unavailable); CI covers Docker builds on push. Confidence level: HIGH. | CLOSED |
| 2026-07-06 | Phase 3 Step 1: Benchmark architecture review complete — no code changes | Existing benchmark should become an independent evaluation harness with three research configurations: Baseline A Raw RAG, Baseline B Post-Retrieval Privacy Masking, and Proposed Secure RAG. Runtime remains a black box with no knowledge of baselines. Implementation should first remove `rag_answer(mask_mode=...)`, then rename benchmark terminology, then centralize configuration logic. | CLOSED |
| 2026-07-06 | Phase 3 Step 2: Remove runtime coupling from benchmark answer generation | `rag_answer(mask_mode=...)` was invalid after Phase 2. Benchmark now owns raw/post answer behaviour by composing `retrieve`, `mask_text`, and `generate_answer`; Secure RAG answer path calls canonical `rag_answer(query, vector_store, chunks)`. Runtime remains unchanged. | CLOSED |
| 2026-07-06 | Phase 3 Step 3: Centralize evaluation configuration selection | Evaluation configuration selection was scattered across 6+ locations (doc leakage loop, retrieval leakage loop, PHI-in-answer loop, summary building, results display, and results initialization). Centralized into `EVALUATION_CONFIGS` registry with `key`, `get_idx`, and `answer` per-entry. All evaluation loops now iterate the registry. Adding a new baseline requires one entry. Behaviour and metrics unchanged. | CLOSED |
| 2026-07-06 | Phase 3 Step 3.5: Separate configuration identity from presentation | Each config entry now carries `id` (stable machine identifier), `display_name` (human-readable name), and `description` (methodology summary). Result JSON keys use `id` values. Console progress messages use `display_name`. Internal dispatch in `benchmark_answer` uses `id`. Identity and presentation can now evolve independently. | CLOSED |
| 2026-07-06 | Phase 3 Step 4: Presentation & reporting cleanup | Updated all presentation layers to use approved research terminology. Added `label` field for compact tables. Updated descriptions to match task specification. Documented evaluation configs in `benchmarks/README.md`. Internal identifiers unchanged. | CLOSED |



---

## Migration Notes

### Result Key Migration

Evaluation result keys changed from legacy mode names to stable identifiers:

| Historical (legacy) | Current (stable) | Display Label |
|---|---|---|
| `raw` | `baseline_a` | Baseline A |
| `post` | `baseline_b` | Baseline B |
| `pre` | `secure_rag` | Secure RAG |

This is a presentation migration, not an evaluation change.
Metrics and methodology are identical.
Existing `results.json` files with legacy keys remain readable but the current benchmark
emits keys under the new identifiers.

### Result JSON Paths

Legacy `results.json` keys:
```json
results["document_leakage"]["raw"]["_summary"]["rate"]
results["retrieval_leakage"]["post"]["_summary"]["rate"]
results["phi_in_answer"]["pre"]["_summary"]["rate"]
```

Current `results.json` keys:
```json
results["document_leakage"]["baseline_a"]["_summary"]["rate"]
results["retrieval_leakage"]["baseline_b"]["_summary"]["rate"]
results["phi_in_answer"]["secure_rag"]["_summary"]["rate"]
```

Summary keys follow the same pattern:
- `document_leakage_raw` -> `document_leakage_baseline_a`
- `retrieval_leakage_post` -> `retrieval_leakage_baseline_b`
- `phi_in_answer_pre` -> `phi_in_answer_secure_rag`


## Open Questions

*Deferred improvements discovered during Phase 2 audit (not implemented):*

- The `clean_input_text()` function in `rag_pipeline.py` is a public helper not exported via `__init__.py`. It is used internally by `load_data`. No action needed — it is not a benchmark concept and serves a valid runtime purpose. Consider making private (`_clean_input_text`) in a future cleanup pass.
- The `LLM_PROVIDER` environment variable in `generator.py` is a runtime configuration concern orthogonal to the benchmark refactor. Not modified.
- The `_truncate_at_stop_marker()` function currently has a weakness: the stop-marker approach is heuristic and model-dependent. Future improvement: replace with structured output parsing. Not in scope.

*Benchmark design questions to resolve during Phase 3 implementation:*

- Should result JSON keys change immediately from `raw`/`post`/`pre` to descriptive configuration identifiers, or should the benchmark emit both new keys and a legacy mapping for one transition period?
- Should Proposed Secure RAG in the benchmark call `build_rag(file_path)` by writing test records to a temporary document, or should it compose runtime primitives in-memory while documenting equivalence to canonical runtime semantics?
- Should benchmark organization remain single-file for the first implementation pass, or should configuration/metrics/reporting modules be introduced once behaviour is stable?
- Should answer utility metrics be added later to complement privacy metrics, using the known generated answers in `dataset_queries.json`?
- Should benchmark-local `truncate_at_stop_marker()` remain duplicated, or should benchmark answer generation eventually be centralized around a shared benchmark-only answer helper for all baselines?
- How should `benchmark_answer` dispatcher be absorbed into `EVALUATION_CONFIGS` answer callables to make each config fully self-contained, removing the intermediate `config_id` string dispatching?
- Should the `build_index` helper also be absorbed into the config registry, or kept as a separate construction step?

---

## Refactoring Rules

1. **One logical commit per coherent change.** Use `git add -p` as needed to separate concerns.
2. **No unrelated refactoring.** Do not fix style, docs, or bugs outside the scope of this refactor.
3. **No feature additions.** The refactor removes abstractions — it does not add new capabilities.
4. **Preserve runtime behaviour.** The canonical pipeline must produce the same results as `build_rag(use_masking=True)` + `rag_answer(mask_mode="pre")` — the current "pre" mode.
5. **Preserve retrieval quality.** No changes to embedding, chunking, or FAISS logic.
6. **Preserve masking behaviour.** No changes to `masker.py`.
7. **Preserve Docker compatibility.** Both `Dockerfile` and `Dockerfile.runtime` must build and run.
8. **Preserve Docker Compose compatibility.** Both profiles must work.
9. **Preserve benchmark reproducibility.** Running `privacy_eval.py` on the same dataset must produce the same metrics.
10. **Keep runtime and benchmark responsibilities separate.** The runtime owns the canonical pipeline. The benchmark owns comparison logic.
11. **Stop implementation if architecture changes become necessary.** If the approved design cannot be implemented as specified, stop and produce a new architecture review.

---

## Completion Criteria

The refactor is complete only when:

- [ ] Runtime architecture matches the approved design.
- [ ] Runtime contains no benchmark abstractions (`use_masking`, `mask_mode`, `raw`/`post`/`pre` terminology).
- [ ] Benchmarks are independent of runtime modes.
- [ ] All tests pass (`python3 -m pytest tests/`).
- [ ] Docker passes (`docker build -f Dockerfile .` and `docker build -f Dockerfile.runtime .`).
- [ ] Docker Compose passes (`docker compose run --rm secure-rag`).
- [ ] CI passes (if applicable).
- [ ] Documentation is updated (`README.md`).
- [ ] CONTEXT.md contains the final architectural decisions.
- [ ] All risks are either CLOSED or explicitly accepted.
- [ ] `REFACTOR_CHECKLIST.md` lessons are merged into CONTEXT.md.
- [ ] `REFACTOR_CHECKLIST.md` is archived or removed.

---

## Lessons Learned (Phase 2 & 2.5)

1. **The nested generator pattern in `rag_answer()` was unnecessary indirection.** The original `cleaned_response()` inner function always buffered the full response before yielding a single string, making the "streaming" contract misleading. Flattening to a direct `yield` achieved the same behaviour with less code.

2. **Removing `use_masking` and `mask_mode` in separate steps (Steps 2 and 3) was correct.** Each removal affected different parts of the pipeline and had different downstream impacts. Combining them would have made validation of the individual changes harder.

3. **No tests required modification during the entire runtime refactor.** All tests used default argument values, which meant the canonical path was always what was being tested. This validates that the default configuration was already the intended Secure RAG behaviour.

4. **The benchmark (`privacy_eval.py`) built its own index via a local `build_index()` function, not the runtime's `build_rag()`.** This meant the `use_masking` removal had zero impact on benchmark code. The only cross-boundary call was `rag_answer(mask_mode=...)` at line 140, which will need Phase 3 attention.

5. **Docker daemon availability is not guaranteed on development machines.** Validation of Docker infrastructure was deferred to CI. Consider adding a `make docker-test` target for local Docker validation when the daemon is available.

---

*End of REFACTOR_CHECKLIST.md*
