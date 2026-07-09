# Retrieval Evaluation Framework — Execution Checklist

> **Status**: Phase 1 — Architecture Review & Design (in progress)
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
| Phase 1 | Architecture Review & Design | IN PROGRESS |
| Phase 2 | Ground Truth Framework | PENDING |
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

### Phase 2 — Ground Truth Framework (future)

- [ ] Define relevance schema: `(qid, record_id, relevant: bool, category: str)`
- [ ] Generate relevance judgments from existing query-record alignment
- [ ] Validate relevance coverage for PHI-targeting and general queries
- [ ] Store ground truth as `benchmarks/retrieval/ground_truth.json`
- [ ] Implement ground truth loader in `_common.py`
- [ ] Validate: all 600 queries have at least one relevant record
- [ ] Phase 2 validation checkpoint

### Phase 3 — Retrieval Runner (future)

- [ ] Design runner interface: `run_retrieval(config, queries, ground_truth, k_values, records, chunks)`
- [ ] Implement retrieval execution for all EVALUATION_CONFIGS
- [ ] Support multiple k values (`1, 3, 5, 10`)
- [ ] Support per-query result storage (retrieved chunk IDs, scores, ranks)
- [ ] Integrate with existing index building (raw / masked)
- [ ] Cache per-config retrieval results for metric computation
- [ ] Validate: runner produces expected top-k results for known-good queries
- [ ] Phase 3 validation checkpoint

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
| `EVALUATION_CONFIGS` registry | `privacy_eval.py` | Yes | Defines 3 configs with `id`, `display_name`, `label`, `get_idx` |
| `load_records()` | `privacy_eval.py` | Yes | Loads `dataset.jsonl` |
| `load_queries()` | `privacy_eval.py` | Yes | Loads `dataset_queries.json` |
| `load_split()` | `privacy_eval.py` | Yes | Loads `train_test_split.json` |
| `build_index()` | `privacy_eval.py` | Yes | Builds raw or masked index from records |
| `retrieve()` | `secure_rag/retriever.py` | Yes | Runtime retrieval: query → top-k chunks |
| `embed_chunks()` | `secure_rag/embedding.py` | Yes | SentenceTransformer embedding |
| `VectorStore` | `secure_rag/vector_store.py` | Yes | FAISS-based similarity search |
| `chunk_text()` | `secure_rag/pdf_loader.py` | Yes | Sliding-window chunking |
| Dataset files | `benchmarks/*.json` / `.jsonl` | Yes | Version-controlled, fixed seeds |

### Missing Infrastructure

| Component | Gap Severity | Notes |
|---|---|---|
| Relevance judgments | HIGH | No explicit mapping of `(query_id → record_id)` — implicitly derivable |
| Chunk→Record mapping | MEDIUM | Chunks are flat strings; no `record_id` attribution |
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
  _common.py          ← Shared utilities (proposed)
  privacy_eval.py     ← Privacy evaluation harness (existing)
  generate_dataset.py ← Dataset generation (existing)
  retrieval/          ← Retrieval evaluation (new)
    __init__.py
    _common.py        ← Retrieval-specific utilities
    ground_truth.py   ← Relevance judgments
    runner.py         ← Retrieval execution
    metrics.py        ← IR metric computation
    failure_analysis.py
    report.py
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
| 1 | No relevance judgments | HIGH | Low | Implicitly derivable: each query's source record is relevant. Generate from existing query-record alignment. |
| 2 | No chunk→record index | MEDIUM | Low | Add `record_id` to chunk metadata during indexing |
| 3 | No IR metrics | HIGH | Low | Implement standard metrics: ~50 LoC each |
| 4 | No retrieval runner | HIGH | Medium | New 200-300 LoC module |
| 5 | No failure classification | MEDIUM | Medium | Taxonomy-driven classifier: ~150 LoC |
| 6 | No shared utility module | LOW | Low | Factor `load_records()`, `load_queries()` into `_common.py` |
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
| R5 | Duplicating privacy eval logic in retrieval eval | Maintenance burden | Medium | Factor shared utilities into `_common.py` before Phase 3 |
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

---

## 9. Validation Log

| Step | Check | Result | Date |
|---|---|---|---|
| — | — | — | — |

*(To be filled during Phase 7)*

---

## 10. Open Questions

1. **Should chunk-level relevance be added alongside record-level?** Record-level is sufficient for the primary research question. Chunk-level enables boundary analysis but requires ground truth per chunk, which is expensive to generate.

2. **Should relevance include graded judgments (0/1/2) or binary only?** Binary is sufficient for Hit Rate, Precision, Recall, MRR. NDCG requires graded. Start with binary; add graded if NDCG becomes necessary.

3. **Should the framework evaluate multi-hop queries (need info from 2+ records)?** Current dataset has single-record queries only. The framework design should support multi-record ground truth, but the initial implementation uses single-record.

4. **How should multi-record queries (future) be handled?** Ground truth schema should support multiple `record_id` entries per query. Runner should aggregate accordingly.

5. **Should retrieval timings be recorded?** Latency is not a research question in this paper. Defer to future work.

6. **Should the framework live in a separate `benchmarks/retrieval/` directory or as `benchmarks/retrieval_eval.py`?** Directory scales better as the framework grows (ground_truth.py, runner.py, metrics.py, failure_analysis.py, report.py). Single-file is acceptable initially but would require refactoring for Phase 5+.

7. **Should retrieval results be cached between runs?** Low priority — dataset is small (120 records, ~600 chunks). Index building is fast.

8. **Should retrieval evaluation run automatically after privacy eval?** Separate scripts, separate invocations. The Makefile or CI can orchestrate both.

9. **What is the evaluation dataset split?** Use existing test split (20 records → 100 queries) for retrieval evaluation. Train split (100 records → 500 queries) available for development.

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

| Metric | Definition | Why |
|---|---|---|
| **Chunk Boundary Recall** | Proportion of relevant chunks that cross a chunk boundary | Quantifies chunk strategy effectiveness |
| **Masking Degradation Ratio** | `metric_secure_rag / metric_baseline_a` per category | Normalized measure of privacy utility tradeoff |
| **Bias@k** | Retrieval performance disparity across demographic groups | Requires dataset annotation beyond current scope |

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
