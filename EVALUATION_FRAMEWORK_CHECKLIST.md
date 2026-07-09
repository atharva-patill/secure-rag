# Retrieval Evaluation Framework — Execution Checklist

> **Status**: Phase 2 — Ground Truth Framework (in progress)
> **Purpose**: Single source of truth for designing, implementing, and validating the Retrieval Evaluation Framework
> **Lifespan**: Temporary — archive or merge into CONTEXT.md after all phases complete

---

## 1. Project Objective

Design and implement a **Retrieval Evaluation Framework** that measures retrieval quality independently from LLM generation. The framework answers research questions such as:

- Does Secure RAG retrieve the correct record?
- Does pre-embedding masking reduce retrieval quality compared to non-masked baselines?
- Which query categories perform well and which fail?
- Why do retrieval failures occur (failure taxonomy)?

The framework becomes a reusable research subsystem within `benchmarks/`.

---

## 2. Design Principles

1. **Runtime agnosticism** — The runtime (`secure_rag/`) remains unaware of the retrieval evaluation framework. Dependency direction is one-way: Retrieval Evaluation → Runtime.

2. **Generation independence** — Retrieval is evaluated independently from LLM generation. Metrics measure what is retrieved, not what is generated from retrieved content.

3. **Record-centric ground truth** — Relevance judgments map `(query_id, record_id)` pairs. Document-level precision precedes chunk-level analysis.

4. **Standard IR metrics first** — Precision@k, Recall@k, MRR@k, Hit Rate@k take precedence over custom or composite metrics.

5. **Failure analysis as first-class artifact** — Every retrieval failure is classified into a taxonomy, enabling systematic research analysis.

6. **Evidence-driven improvements** — Failure patterns should directly suggest future retrieval improvements (embedding choice, chunking strategy, query expansion).

7. **Extensibility** — The framework should support future retrieval research beyond this paper: different embedding models, chunk strategies, retrieval algorithms, and query categories.

8. **Reproducibility** — Fixed seeds, deterministic execution, version-controlled dataset, cached indices.

---

## 3. Overall Roadmap

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Architecture Review & Design | COMPLETE |
| Phase 2 | Ground Truth Framework | COMPLETE |
| Phase 3 | Retrieval Runner | PENDING |
| Phase 4 | IR Metrics | PENDING |
| Phase 5 | Failure Analysis | PENDING |
| Phase 6 | Reporting & Visualization | PENDING |
| Phase 7 | Research Validation | PENDING |
| Phase 8 | Documentation | PENDING |
| Phase 9 | CONTEXT.md Update | PENDING |

---

## 4. Phase Progress

### Phase 1 — Architecture Review & Design (this document)

| Step | Description | Status |
|---|---|---|
| 1.1 | Audit current benchmark capabilities | COMPLETE |
| 1.2 | Audit existing reusable components | COMPLETE |
| 1.3 | Define evaluation philosophy | COMPLETE |
| 1.4 | Define ground truth philosophy | COMPLETE |
| 1.5 | Design repository architecture | COMPLETE |
| 1.6 | Verify dependency boundary | COMPLETE |
| 1.7 | Design evaluation pipeline | COMPLETE |
| 1.8 | Identify all evaluation components | COMPLETE |
| 1.9 | Define metric strategy | COMPLETE |
| 1.10 | Design failure taxonomy | COMPLETE |
| 1.11 | Define research methodology | COMPLETE |
| 1.12 | Review and refine implementation roadmap | COMPLETE |
| 1.13 | Gap analysis | COMPLETE |
| 1.14 | Risk register | COMPLETE |
| 1.15 | Decision log | COMPLETE |
| 1.16 | Produce deliverables | COMPLETE |

### Phase 2 — Ground Truth Framework (complete)

- [x] Define relevance schema: `(qid, record_id, relevant: bool, category: str, subcategory: str, expected_behaviour: str)`
- [x] Generate relevance judgments from existing query-record alignment
- [x] Validate relevance coverage for PHI-targeting and general queries
- [x] Store ground truth as `benchmarks/retrieval/ground_truth_v1.json`
- [x] Validate: all 600 queries have at least one relevant record
- [x] Validate: all 120 referenced records exist in dataset
- [x] Validate: all categories are valid (general, phi_targeting)
- [x] Validate: all expected_behaviour values are valid (record_retrieval, entity_retrieval)
- [x] Validate: no orphan queries or orphan records
- [x] Validate: ground truth generation is deterministic (two runs produce identical results)
- [x] Factor shared utilities into `benchmarks/_common.py`
- [x] Refactor `benchmarks/privacy_eval.py` to import from `_common.py` (eliminates duplication)
- [x] Phase 2 validation checkpoint — PASS
- [x] D8: Ground truth includes query categorization (category + subcategory) and expected behaviour.

### Phase 3 — Retrieval Runner (future)

- [ ] Design runner interface: `run_retrieval(config, queries, ground_truth, k_values, records, chunks)`
- [ ] Implement retrieval execution for all EVALUATION_CONFIGS
- [ ] Support multiple k values (`1, 3, 5, 10`)
- [ ] Build chunk→record index so retrieved chunks can be mapped to ground truth records
- [ ] Support per-query result storage (retrieved chunk IDs, scores, ranks)
- [ ] Integrate with existing index building (raw / masked)
- [ ] Cache per-config retrieval results for metric computation
- [ ] Validate: runner produces expected top-k results for known-good queries
- [ ] Phase 3 validation checkpoint
- [ ] D9: Retrieval runner outputs become the canonical evaluation artifact.
Reason:
Metrics, failure analysis, visualization, and validation all consume the same retrieval results.

### Phase 4 — IR Metrics (future)

- [ ] Implement `hit_rate_at_k(retrieved, ground_truth, k)`
- [ ] Implement `precision_at_k(retrieved, ground_truth, k)`
- [ ] Implement `recall_at_k(retrieved, ground_truth, k)`
- [ ] Implement `mrr_at_k(retrieved, ground_truth, k)`
- [ ] Implement metric aggregator: mean and per-query metrics
- [ ] Implement per-category metric breakdown (PHI-targeting vs general)
- [ ] Validate against known test cases (e.g., k=5, same config → expected scores)
- [ ] Phase 4 validation checkpoint

### Phase 5 — Failure Analysis (future)

- [ ] Implement failure classifier based on taxonomy (see Section 11)
- [ ] Classify each retrieval failure by category
- [ ] Compute per-category failure rates
- [ ] Identify systematic degradation patterns (masking, chunk boundaries, embedding)
- [ ] Generate failure report with examples
- [ ] Validate: manual review of classified failures for correctness
- [ ] Phase 5 validation checkpoint
- [ ] D10 Failure analysis remains an independent phase.
Reason:
Metrics explain how well retrieval performs; failure analysis explains why it performs that way. Keeping them separate mirrors the separation between quantitative evaluation and qualitative analysis found in research papers.

### Phase 6 — Reporting & Visualization (future)

- [ ] Design results JSON schema for retrieval metrics
- [ ] Implement console summary with per-k, per-config, per-category tables
- [ ] Integrate retrieval metrics into existing `results.json`
- [ ] Ensure summary keys are distinct from privacy eval keys
- [ ] Validate: report matches raw metric computation
- [ ] Phase 6 validation checkpoint

### Phase 7 — Research Validation (future)

- [ ] Functional validation: all 3 configs produce retrieval metrics
- [ ] Metric validation: expected relationships hold (raw ≥ post ≥ Secure RAG in recall)
- [ ] Reproducibility validation: two consecutive runs produce identical results
- [ ] Methodology audit: same dataset, same retrieval, same metrics across configs
- [ ] Fair comparison audit: only index masking differs
- [ ] Runtime boundary audit: no retrieval evaluation code in `secure_rag/`
- [ ] Phase 7 validation checkpoint

### Phase 8 — Documentation (future)

- [ ] Update `benchmarks/README.md` with retrieval evaluation section
- [ ] Document ground truth schema
- [ ] Document metric definitions
- [ ] Document failure taxonomy
- [ ] Document CLI usage
- [ ] Document interpretation guidelines
- [ ] Phase 8 validation checkpoint

### Phase 9 — CONTEXT.md Update (future)

- [ ] Merge final architecture decisions into CONTEXT.md
- [ ] Archive `EVALUATION_FRAMEWORK_CHECKLIST.md`
- [ ] Update repository architecture diagram
- [ ] Phase 9 validation checkpoint

---

## 5. Architecture Review

### Current State

**Runtime** (`secure_rag/`):
- Canonical pipeline: load → clean → split → mask (mandatory) → chunk → embed → index → retrieve → generate
- Two entry points: `build_rag(file_path)` and `rag_answer(query, vector_store, chunks)`
- No benchmark concepts, no modes, no optional behaviour (validated in Phase 2/3 of runtime refactor)

**Benchmark** (`benchmarks/`):
- `privacy_eval.py` — evaluation harness with `EVALUATION_CONFIGS` registry (Baseline A, Baseline B, Secure RAG)
- `generate_dataset.py` — 120 synthetic Indian medical records + 600 queries (360 PHI-targeting, 240 general)
- `results.json` — structured output with per-metric, per-config data
- Evaluates: document leakage, retrieval leakage, masking recall, PHI in answers
- Does NOT evaluate retrieval quality (precision, recall, ranking)

**Dataset structure** (critical for retrieval eval):
- 120 records, each with `record_id` (e.g., `MED117`, `MED070`, etc.)
- 600 queries, each with: `qid`, `question`, `answer`, `phi_in_answer` flag
- Each query has implicit parent record (the record it belongs to in the file)
- 5 queries per record (3 PHI-targeting + 2 general)

### Existing Reusable Components

| Component | Location | Reusable? | Notes |
|---|---|---|---|
| `EVALUATION_CONFIGS` registry | `_common.py` (moved) | Yes | Defines 3 configs with `id`, `display_name`, `label`, `get_idx` |
| `load_records()` | `_common.py` (moved) | Yes | Loads `dataset.jsonl` |
| `load_queries()` | `_common.py` (moved) | Yes | Loads `dataset_queries.json` |
| `load_split()` | `_common.py` (moved) | Yes | Loads `train_test_split.json` |
| `build_index()` | `_common.py` (moved) | Yes | Builds raw or masked index from records |
| Ground truth | `benchmarks/retrieval/ground_truth_v1.json` | Yes | 600 entries with relevance, category, subcategory, expected behaviour |
| Ground truth API | `benchmarks/retrieval/ground_truth.py` | Yes | `generate_ground_truth()`, `load_ground_truth()`, `validate_ground_truth()`, `save_ground_truth()` |
| `retrieve()` | `secure_rag/retriever.py` | Yes | Runtime retrieval: query → top-k chunks |
| `embed_chunks()` | `secure_rag/embedding.py` | Yes | SentenceTransformer embedding |
| `VectorStore` | `secure_rag/vector_store.py` | Yes | FAISS-based similarity search |
| `chunk_text()` | `secure_rag/pdf_loader.py` | Yes | Sliding-window chunking |
| Dataset files | `benchmarks/*.json` / `.jsonl` | Yes | Version-controlled, fixed seeds |

### Missing Infrastructure

| Component | Gap Severity | Notes |
|---|---|---|
| Relevance judgments | HIGH → **CLOSED (Phase 2)** | `benchmarks/retrieval/ground_truth_v1.json` with 600 entries |
| Chunk→Record mapping | MEDIUM | Chunks are flat strings; no `record_id` attribution — deferred to Phase 3 |
| IR metrics | HIGH | No precision@k, recall@k, MRR@k, hit rate |
| Retrieval runner | HIGH | No dedicated retrieval evaluation loop |
| Failure classifier | MEDIUM | No systematic failure taxonomy implementation |
| Retrieval report | MEDIUM | No retrieval-specific console/JSON reporting |
| Shared utility module | LOW | Loaders duplicated between privacy and retrieval eval |
| Variable k support | LOW | `k=5` is hardcoded in privacy eval |

### Architectural Boundaries

```
secure_rag/         ← Runtime (no knowledge of evaluation)
  rag_pipeline.py
  retriever.py
  embedding.py
  vector_store.py
  generator.py
  masker.py
  pdf_loader.py
  cli.py
  __init__.py

benchmarks/
  _common.py          ← Shared utilities (loaders, config registry, helpers)
  privacy_eval.py     ← Privacy evaluation harness (imports from _common.py)
  generate_dataset.py ← Dataset generation (unchanged)
  retrieval/          ← Retrieval evaluation framework (new)
    __init__.py       ← Package init
    ground_truth.py   ← Ground truth generation & validation (Phase 2 — COMPLETE)
    ground_truth_v1.json  ← Generated ground truth data
    runner.py         ← Retrieval execution (Phase 3 — future)
    metrics.py        ← IR metric computation (Phase 4 — future)
    failure_analysis.py   (Phase 5 — future)
    report.py             (Phase 6 — future)
  dataset.jsonl
  dataset_queries.json
  train_test_split.json
  results.json
  README.md
```

Direction: `retrieval/` → `privacy_eval.py` (for `EVALUATION_CONFIGS`) → `secure_rag/` (for runtime primitives). No reverse dependencies.

---

## 6. Gap Analysis

| # | Gap | Severity | Effort | Mitigation |
|---|---|---|---|---|
| 1 | No relevance judgments | CLOSED | Low | Ground truth v1 generated in Phase 2. 600 queries, 120 records, 5 categories, 2 behaviours. |
| 2 | No chunk→record index | MEDIUM | Low | Add `record_id` to chunk metadata during indexing — deferred to Phase 3 |
| 3 | No IR metrics | HIGH | Low | Implement standard metrics: ~50 LoC each |
| 4 | No retrieval runner | HIGH | Medium | New 200-300 LoC module |
| 5 | No failure classification | MEDIUM | Medium | Taxonomy-driven classifier: ~150 LoC |
| 6 | No shared utility module | CLOSED | Low | `benchmarks/_common.py` created in Phase 2. Loaders + config registry + utility functions extracted from `privacy_eval.py`. |
| 7 | No variable k support | LOW | Low | Parameterize `k` in runner |
| 8 | No retrieval-specific reporting | MEDIUM | Low | New report module extending existing patterns |

---

## 7. Risk Register

| # | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| R1 | Chunk boundary splits relevant answer across chunks | Inflated precision (missed relevant content) | Medium | Use record-level relevance; chunk-level is secondary |
| R2 | Masking destroys tokens needed for entity-specific retrieval | Expected low recall for Secure RAG on PHI queries | High (expected) | This IS the research question — measure the degradation, treat as finding |
| R3 | Existing queries designed for privacy, not retrieval utility | Queries may not test ranking quality | Low | 240 general queries test factual retrieval; 360 PHI queries test entity retrieval |
| R4 | Ground truth is single-record (one relevant per query) | Cannot test multi-record retrieval | Low | Current dataset limitation; document as assumption; extensible to multi-record in future |
| R5 | Duplicating privacy eval logic in retrieval eval | Maintenance burden | Medium | **CLOSED** — `benchmarks/_common.py` factored in Phase 2. Both privacy and retrieval eval consume the same shared module. |
| R6 | Results.json key collision between privacy and retrieval eval | Overwritten or ambiguous keys | Low | Use distinct key prefixes (`retrieval_precision_k5`, etc.) |
| R7 | Retrieval runner changes index-building assumptions | Privacy eval metrics shift | Low | Retrieval eval reuses existing `build_index()` — no index changes |

---

## 8. Decision Log

| # | Date | Decision | Reason | Status |
|---|---|---|---|---|
| D1 | — | Ground truth is **record-centric** (question → record) | Dataset structure: each query belongs to exactly one record. Record-level evaluation maps to research question "does the system find the right patient record?" Chunk-level is secondary for boundary analysis. | OPEN |
| D2 | — | Retrieval evaluation lives in **`benchmarks/retrieval/`** | New directory sibling to evaluation modules, not merged into `privacy_eval.py`. Separation of concerns: privacy vs. utility evaluation. Shared infrastructure factored into `_common.py`. | OPEN |
| D3 | — | Metrics computed **per-config** (Baseline A, B, Secure RAG) | Consistent with existing evaluation architecture. Directly answers "does masking degrade retrieval?" and isolates the embedding effect. | OPEN |
| D4 | — | Standard IR metrics take precedence over custom metrics | Precision@k, Recall@k, MRR@k, Hit Rate@k are well-understood, comparable to literature. Custom metrics are secondary and must be validated against standard ones. | OPEN |
| D5 | — | Failure analysis is a **first-class research artifact** | Not an ad-hoc debugging step. Taxonomy-driven classification produces systematic degradation patterns. | OPEN |
| D6 | — | Queries are never masked in any configuration | Consistent with existing design principle. Ensures fair comparison: only the index differs. | OPEN |
|---|---|---|---|---|---|
| D7 | 2026-07-09 | Factor shared utilities into `benchmarks/_common.py` | Avoids duplication between privacy_eval.py and retrieval evaluation. Both import loaders, configs, and utility functions from the same module. | CLOSED |
| D8 | 2026-07-09 | Ground truth includes per-query category + subcategory + expected behaviour | Enables per-category metric breakdown and failure analysis without regenerating ground truth. Categories: general/phi_targeting. Subcategories: factual_hospital, summary, phi_aadhaar, phi_phone, phi_mrn. Behaviours: record_retrieval, entity_retrieval. | CLOSED |
| D9 | 2026-07-09 | Ground truth is versioned (`v1`, `v2`, etc.) | Allows forward evolution without breaking downstream phases. Each version is a separate file. Downstream phases pin to specific version. | CLOSED |
| D10 | 2026-07-09 | Ground truth is generated by a standalone script (`ground_truth.py`), not embedded in the runner | Decouples ground truth generation from retrieval execution. Runner consumes the static file. Ground truth can be updated independently. | CLOSED |

---

## 9. Validation Log

| Step | Check | Result | Date |
|---|---|---|---|---|
| P2-V1 | Every query has ground truth | 600/600 queries have `relevant_records` | 2026-07-09 |
| P2-V2 | Every referenced record exists | 120/120 referenced `record_id` values exist in `dataset.jsonl` | 2026-07-09 |
| P2-V3 | Every query has a valid category | `general` (240) + `phi_targeting` (360) — all valid | 2026-07-09 |
| P2-V4 | No orphan records | All 120 records referenced by at least one query | 2026-07-09 |
| P2-V5 | No orphan queries | All 600 queries have a non-empty `relevant_records` | 2026-07-09 |
| P2-V6 | Ground truth generation is deterministic | Two consecutive runs produce identical JSON (excluding timestamp) | 2026-07-09 |
| P2-V7 | Repeated generation produces identical results | Confirmed: `d1 == d2` after timestamp removal | 2026-07-09 |
| P2-V8 | 42 runtime tests still pass | 42/42 passed before Phase 2, 42/42 after | 2026-07-09 |

---

## 10. Open Questions

1. **Should chunk-level relevance be added alongside record-level?** **ANSWERED (Phase 2):** Record-level is sufficient for Phase 1 research. Chunk-level boundary analysis is deferred to Phase 5 (Failure Analysis), where it will be computed from chunk→record mapping rather than explicit per-chunk judgments.

2. **Should relevance include graded judgments (0/1/2) or binary only?** **ANSWERED (Phase 2):** Binary. Ground truth v1 uses binary relevance. NDCG is deferred to "Future" metric category.

3. **Should the framework evaluate multi-hop queries (need info from 2+ records)?** **ANSWERED (Phase 2):** Not in v1. Ground truth schema supports multiple `record_id` entries per query (`relevant_records: list[str]`), enabling future extension without schema changes.

4. **How should multi-record queries (future) be handled?** **ANSWERED (Phase 2):** Schema already supports this via `relevant_records: list[str]`. Runner will aggregate metrics accordingly when multi-record queries exist.

5. **Should retrieval timings be recorded?** Deferred. Latency is not a research question in this paper.

6. **Should the framework live in a separate `benchmarks/retrieval/` directory or as `benchmarks/retrieval_eval.py`?** **ANSWERED (Phase 2):** Directory structure `benchmarks/retrieval/` was chosen. Currently contains `ground_truth.py`. Future phases add `runner.py`, `metrics.py`, `failure_analysis.py`, `report.py`.

7. **Should retrieval results be cached between runs?** Deferred. Small dataset (~600 chunks); index building is fast.

8. **Should retrieval evaluation run automatically after privacy eval?** Deferred. Separate scripts, separate invocations. CI can orchestrate both.

9. **What is the evaluation dataset split?** Use existing test split (20 records → 100 queries). Train split (100 records → 500 queries) for development.

---

## 11. Failure Taxonomy

```
Retrieval Failure
│
├── Entity Retrieval Failure          (query specifies a named entity)
│   ├── Diagnosis                     e.g., "Patient diagnosed with Typhoid"
│   ├── Hospital / Clinic             e.g., "Which hospital did the patient visit?"
│   ├── Medication                    e.g., "What medication was prescribed?"
│   ├── Treatment / Procedure         e.g., "What treatment was administered?"
│   ├── Patient Demographics          e.g., patient age, gender, blood type
│   └── Contact / Identifier          e.g., phone, email, Aadhaar, PAN, MRN
│
├── General Query Failure             (query asks for summary/factual information)
│   ├── Summary / Overview            e.g., "Summarize the patient record"
│   └── Factual / Attribute           e.g., "Who was the attending physician?"
│
├── Ranking Failure                   (relevant record retrieved but not at top-k)
│   ├── Low Similarity Score          relevant record scored below threshold
│   └── Competitor Record Higher      another record's chunk scored higher
│
├── Chunk Boundary Failure            (relevant content split across chunks)
│   ├── Answer Fragmented             required information divided by chunk boundary
│   └── Context Lost                  chunk truncation removes key content
│
├── Masking Degradation               (pre-embedding masking removes retrieval tokens)
│   ├── Entity Name Masked            e.g., "[NAME_MASKED]" loses "Rajesh Kumar"
│   ├── Location Masked               e.g., "[LOC_MASKED]" loses "AIIMS Delhi"
│   └── Identifier Masked             e.g., "[PHONE_MASKED]" loses contact info
│
└── Embedding Similarity Failure      (semantic mismatch between query and record)
    ├── Vocabulary Mismatch           query uses different terms than record
    └── Domain Drift                  embedding model unfamiliar with medical terms
```

---

## 12. Metric Strategy

### Mandatory (Phase 4)

| Metric | Definition | Why |
|---|---|---|
| **Hit Rate@k** | Proportion of queries where the relevant record is in the top-k retrieved chunks | Primary retrieval success metric. Answers "did the system find the right record?" |
| **MRR@k** | Mean Reciprocal Rank: average of `1 / rank_of_first_relevant` across queries | Measures ranking quality. Penalizes failures where relevant is at position 6+. |
| **Precision@k** | Proportion of retrieved chunks that belong to the relevant record | Measures how much of the retrieval result is useful. Secondary to Hit Rate. |
| **Recall@k** | Proportion of relevant chunks that were retrieved (out of all chunks of the relevant record) | Measures coverage of the relevant record's content. Important for assessing fragmentation. |

### Future

| Metric | Definition | Why |
|---|---|---|
| **NDCG@k** | Normalized Discounted Cumulative Gain | Requires graded relevance. Useful if we add multi-record or partial-relevance judgments. |
| **Per-category breakdown** | Hit Rate / MRR computed separately for PHI-targeting vs. general queries | Answers "do specific query types perform better/worse?" |

### Experimental

| Metric                        | Definition                                                | Why                                              |
| -------------------------------| -----------------------------------------------------------| --------------------------------------------------|
| **Chunk Boundary Recall**     | Proportion of relevant chunks that cross a chunk boundary | Quantifies chunk strategy effectiveness          |
| **Masking Degradation Ratio** | `metric_secure_rag / metric_baseline_a` per category      | Normalized measure of privacy utility tradeoff   |
| **Bias@k**                    | Retrieval performance disparity across demographic groups | Requires dataset annotation beyond current scope |

---

## 13. Research Methodology

### Framework in Paper

```
Section 4.1 — Privacy Evaluation (existing)
  │
  │   Document Leakage: PII in indexed chunks
  │   Retrieval Leakage: PII in retrieved chunks
  │   Masking Recall:   masker coverage
  │   PHI in Answers:   PII in generated answers
  │
  ▼
Section 4.2 — Retrieval Evaluation (proposed)
  │
  │   Hit Rate@k, MRR@k: "does the system find the correct record?"
  │   Per-config comparison: Baseline A vs B vs Secure RAG
  │   Per-category analysis: PHI-targeting vs general queries
  │
  ▼
Section 4.3 — Privacy–Utility Tradeoff (synthesis)
  │
  │   Joint table: privacy leakage rates + retrieval success rates
  │   Tradeoff analysis: Baseline B masks after retrieval (no retrieval cost),
  │   Secure RAG masks before indexing (retrieval cost for entity queries)
  │
  ▼
Section 4.4 — Failure Analysis (qualitative)
  │
  │   Failure taxonomy distribution per config
  │   Representative failure examples
  │   Root cause analysis: masking degradation, chunk boundaries
  │
  ▼
Section 5 — Discussion & Implications
      │
      │   Research findings
      │   Recommendations for practitioners
      │   Limitations
      │   Future work
```

---

## Implementation Rules

1. **One logical commit per coherent change.** Use `git add -p` as needed to separate concerns.
2. **No unrelated changes.** Do not fix style, docs, or bugs outside the scope of this framework.
3. **No runtime modifications.** `secure_rag/` files must not be changed.
4. **No benchmark regression.** Existing `privacy_eval.py` must continue to produce identical results.
5. **Preserve dataset integrity.** Ground truth must be derivable, not guessed.
6. **Stop if architecture changes become necessary.** Produce a new architecture review before proceeding.
7. **Validate before moving to next phase.** Each phase has a validation checkpoint.

---

## Completion Criteria

The Retrieval Evaluation Framework is complete only when:

- [ ] All 9 phases are complete and validated.
- [ ] Retrieval metrics are computed for all 3 configurations (Baseline A, B, Secure RAG).
- [ ] Retrieval metrics are reported per-k (1, 3, 5, 10).
- [ ] Retrieval metrics are broken down per-category (PHI-targeting vs general).
- [ ] Failure classification is applied and reported.
- [ ] `benchmarks/README.md` documents the framework.
- [ ] `CONTEXT.md` contains final architectural decisions.
- [ ] `EVALUATION_FRAMEWORK_CHECKLIST.md` is archived.
- [ ] All risks are either CLOSED or explicitly accepted.
- [ ] 42/42 runtime tests pass, and retrieval evaluation produces reproducible results.
