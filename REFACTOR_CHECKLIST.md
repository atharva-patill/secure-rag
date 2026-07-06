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
| Phase 2 | Runtime Refactor | IN PROGRESS (Step 1: Pipeline simplification — complete; Step 2: Optional masking removal — complete; Step 3: Query mode removal — complete) |
| Phase 2.5 | Validation | NOT STARTED |
| Phase 3 | Benchmark Refactor | NOT STARTED |
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

### Remaining Steps

- [ ] Runtime API simplified (see [Public Runtime API Philosophy](#public-runtime-api-philosophy)).
- [ ] Runtime architecture matches the approved design.
- [ ] CLI still works.
- [ ] Docker still builds.
- [ ] Docker Compose still works.
- [ ] Runtime tests pass.

---

## Validation Matrix

### Runtime

- [ ] Package imports (`from secure_rag import build_rag, rag_answer`)
- [ ] Python API (`build_rag("file.txt")` returns `(VectorStore, List[str])`)
- [ ] Python API (`rag_answer("query", vs, chunks)` returns generator)
- [ ] CLI (`secure-rag data.txt` starts and responds to queries)

### Masking

- [ ] Pre-embedding masking is applied in `build_rag()`
- [ ] Record segmentation preserves blank-line boundaries
- [ ] Chunking produces correct overlapping windows
- [ ] Retrieval returns expected chunks

### Infrastructure

- [ ] Docker build (`docker build -f Dockerfile .`)
- [ ] Docker build (`docker build -f Dockerfile.runtime .`)
- [ ] Docker Compose (`docker compose run --rm secure-rag`)
- [ ] Docker Compose with Ollama profile (`docker compose --profile ollama up`)

### CI

- [ ] Python tests pass (`python3 -m pytest tests/`)
- [ ] Docker CI passes (if applicable)

---

## Benchmark Refactor Checklist

- [ ] Raw baseline implemented independently (compose runtime primitives directly)
- [ ] Post-retrieval masking baseline implemented independently (compose runtime primitives + `mask_text`)
- [ ] Secure RAG baseline uses canonical runtime (`build_rag` + `rag_answer` with no modes)
- [ ] Benchmark no longer depends on runtime modes (`mask_mode` argument removed)
- [ ] Metrics unchanged (document leakage, retrieval leakage, masking recall, PHI in answers)
- [ ] Evaluation results reproducible (`python3 benchmarks/privacy_eval.py`)

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
| 2 | Benchmark depends on removed `mask_mode` parameter | `privacy_eval.py:140` will raise `TypeError` at runtime — `rag_answer()` no longer accepts `mask_mode` | Refactor benchmark in Phase 3 to call `rag_answer()` without `mask_mode` and implement post-masking locally | OPEN — partially actualized |
| 3 | Documentation drift after refactor | README describes removed modes/parameters | Update README in same PR as runtime changes | OPEN |
| 4 | Runtime behaviour accidentally changes | Masking is skipped or applied incorrectly | Run full test suite + benchmark validation | OPEN |
| 5 | Tests become inconsistent with new API | Tests fail due to changed signatures | Update tests in same pass as runtime changes | OPEN |
| 6 | Docker build regression | Image fails due to import errors | Run `docker build` after refactor | OPEN |
| 7 | Docker Compose regression | Compose services fail to start | Run `docker compose run --rm secure-rag` | OPEN |
| 8 | CI regression | Pipeline fails after merge | Run `pytest` before committing | OPEN |
| 9 | Public imports breaking | `from secure_rag import build_rag` fails | Update `__init__.py` exports | OPEN |
| 10 | Benchmark reproducibility changing | Different results from same dataset | Run `privacy_eval.py` before and after to compare | OPEN |
| 11 | Metrics changing unexpectedly | Privacy metrics shift due to unintended behaviour change | Validate baseline metrics match before/after | OPEN |
| 12 | Runtime and benchmark diverging after refactor | Benchmark no longer tests the actual runtime pipeline | Keep Secure RAG mode in benchmark using canonical runtime | OPEN |

---

## Decision Log

| Date | Decision | Reason | Status |
|---|---|---|---|---|
| 2026-07-06 | Step 1: Extract `_truncate_at_stop_marker()` and flatten `rag_answer` generator | Improve architectural clarity without changing observable behaviour. Nested `cleaned_response()` added unnecessary indirection. Extraction gives response post-processing a clear name and separates concerns. | CLOSED |
| 2026-07-06 | Step 2: Remove `use_masking` parameter from `build_rag()`, make `mask_text()` unconditional | The approved architecture defines Secure RAG as a deterministic runtime. Optional masking exists only for benchmark baselines and has no place in the runtime. `build_rag()` now takes only `file_path`. No behavioural change for the canonical Secure RAG path. | CLOSED |
| 2026-07-06 | Step 3: Remove `mask_mode` parameter from `rag_answer()`, remove `raw`/`post`/`pre` terminology | The approved architecture defines exactly one query pipeline with no modes. The benchmark will implement raw/post baselines independently in Phase 3. `rag_answer()` now takes only `(query, vector_store, chunks)`. | CLOSED |

---

## Open Questions

*None yet.*

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

## Lessons Learned

*To be filled after refactor completion.*

---

*End of REFACTOR_CHECKLIST.md*
