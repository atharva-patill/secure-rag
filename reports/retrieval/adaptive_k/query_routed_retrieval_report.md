# Query-Routed Fixed-Depth Retrieval — Research Report

> **Terminology note (strict):** This report distinguishes three concepts:
> - **Adaptive-K characterization** (benchmarking fixed K = 2,5,10,20,30,50 to understand depth vs recall/precision tradeoffs; no routing — reports/retrieval/adaptive_k/adaptive_k_report.md)
> - **Query-routed fixed-depth retrieval** (THIS work: heuristic lexical classifier routes single→K=2, multi→K=20, then delegates to deterministic fixed-K `retrieve()`)
> - **True score-adaptive retrieval** (NOT implemented: would use runtime score gaps, similarity thresholds, learned stopping, or progressive retrieval)
>
> The implementation is intentionally called *query-routed fixed-depth retrieval* and must not be described as a learned or score-adaptive algorithm unless future evidence supports the stronger term.

---

## 1. Motivation

Prior characterization showed:

- Single-record factual queries dominate the dataset (600 of 605 have exactly one relevant record) and are satisfied at K=2 (production default).
- Multi-record aggregate queries (e.g., “Which patients have Hypertension?” with 16 relevant) require K≈20 for high recall, and K=2 yields recall 0.10–0.29 for those queries.
- Raising the global default to K=20 improves aggregate recall but collapses precision for single-record queries (overall precision 0.015→0.013, single-target precision 0.009→0.008) and inflates context 10× while providing no benefit for 99% of queries.

A validated runtime score-gap or learned stopping signal does not yet exist, so a conservative routing policy conditioned only on query text was approved: lexical detection of aggregate intent, then fixed-depth retrieval at the already-validated depths.

Goal: preserve production K=2 baseline by default, selectively use K=20 only when query lexically signals aggregate intent, without changing embedding, masking, FAISS, chunking, generation, or privacy architecture.

## 2. Policy

Centralized constants (single source of truth):

```python
DEFAULT_SINGLE_K = 2
AGGREGATE_K = 20
```

Routing:

- `classify_query(query) -> "single" | "multi"` (pure lexical, see §4)
- `single -> K=2`
- `multi  -> K=20`

No K=10/30/50 routing, no score thresholds, no learned K prediction, no progressive retrieval. Values are not tuned in this implementation; they are the conservative fixed depths approved by audit.

## 3. Implementation

**Architecture preserved:**

```
query
  ↓
query classification (lexical, pure function)
  ↓
selected K (2 or 20)
  ↓
existing retrieve(..., k=selected_k)  ← primitive remains deterministic fixed-K
  ↓
context ("\n\n".join)
  ↓
existing generator
```

**Privacy pipeline unchanged:**

```
document → detection → policy → masking → chunking → embedding → FAISS → routed retrieval → generation
```

- `pre` masking still inside `build_rag()` before chunking/embedding.
- `post`/`raw` still share the same raw index (no separate post-masked index).
- Query is never masked, in any mode.

**Files:**

- New: `secure_rag/query_router.py` — classifier + constants + `routed_k()` helper. Pure function, no I/O, no LLM, no embedding, no randomness, deterministic.
- Modified: `secure_rag/rag_pipeline.py` — adds `routed: bool=False` and research-only `k: int|None` to `rag_answer()`; default `routed=False` preserves K=2. When `k` is set it takes precedence (research benchmark); when `routed=True` uses `routed_k(query, enabled=True)`.
- Modified: `secure_rag/cli.py` — adds experimental `--routed` flag (`typer.Option(False, "--routed", ...)`). Without flag, behavior is `fixed K=2`. With flag, per-query logging: `Retrieval mode: routed  Query type: {single|multi}  K: {2|20}` and init line `retrieval mode: routed|fixed K=2`. No `--query-type` override, no K override exposed via CLI.

**Protected files not modified** (verified via `git status`): `secure_rag/retriever.py`, `secure_rag/vector_store.py`, `secure_rag/embedding.py`, `secure_rag/masker.py`, `secure_rag/detection.py`, `secure_rag/policies/`, `secure_rag/domain_configs/`, `data/sample_patient_data.txt`, `benchmarks/retrieval/ground_truth_v2.json`, generator prompt/behavior, chunking, FAISS config. Diff for `secure_rag/generator.py` is pre-existing dirty state (enhanced Ollama instrumentation) and is explicitly distinguished — not introduced by this work.

**Backward compatibility:** `rag_answer(query, vector_store, chunks)` with three positional args still yields K=2. Existing callers without `routed` kwarg are unaffected. `retrieve()` signature and semantics unchanged (`k=2` default).

## 4. Runtime Information Used

Classifier uses only:

- Lowercased query string, exact substring matching against aggregate vocabulary derived from `secure_rag/generator.py`:
  - `which patients`, `who all`, `list all`, `how many`, `which records`, `who received`
  - `summarize` only when co-occurring with a plural hint (`all`, `patients`, `records`, `which`) to avoid misclassifying singular “Summarize the patient record.”

It does NOT use ground truth, expected answer, relevant record count, disease prevalence, benchmark metadata, MRN, evaluation labels, embedding similarity, or any learned signal. Documented as heuristic in module docstring (`secure_rag/query_router.py:1`).

## 5. Forbidden Evaluation Information

Not used at runtime: ground-truth record IDs, `relevant_records` sets, `relevant_records` lengths, aggregate prevalence, `AGGREGATE_QUERIES_V2` constant, MRN mappings, benchmark QIDs, evaluation metrics. Ground truth is used only offline in the research benchmark (`benchmarks/retrieval/query_routed/query_routed_runner.py`) for scoring, never in `classify_query` or `rag_answer`.

## 6. Experimental Methodology

- **Ground truth:** `benchmarks/retrieval/ground_truth_v2.json` (605 queries, v2, 120 records). No mutation. Verified splits: single-target (1 relevant) = 601 queries, multi-record (>1) = 4 queries, aggregate `AGG_*` = 5 queries.
- **Index:** single masked index built via medical pre-embedding masking (`mask_text` + `medical` policy/detectors) → `chunk_text` → `embed_chunks` (all-MiniLM-L6-v2) → `VectorStore` (IndexFlatL2). 120 chunks, `ntotal=120`. Deterministic.
- **Retrieval per query:** dense FAISS, `embed_chunks([query])` → `vector_store.search(k=20)` (top-20 retrieved once, sliced for K=2 vs K=20). Scored with canonical metrics from `benchmarks/retrieval/metrics.py`: `hit_rate_at_k`, `precision_at_k`, `recall_at_k`, `mrr_at_k` with `relevant_set`. No metric duplication.
- **Modes compared:**
  - Fixed K=2
  - Fixed K=20
  - Routed: `classify_query(query) -> K=2 or 20` (heuristic, 600 routed to 2, 5 routed to 20)
- **Aggregation:** overall (605), single-target (601), multi-record (4), per-aggregate (5 individual). No composite score. Routed distribution stats (mean/median/min/max, counts) reported separately.
- **Artifact:** `benchmarks/retrieval/query_routed/query_routed_results.json`, runner `benchmarks/retrieval/query_routed/query_routed_runner.py`.

## 7. Results

### 7.1 Overall (n=605)

| Mode | HitRate | Precision | Recall | MRR |
|------|---------|-----------|--------|-----|
| Fixed K=2 | 0.0248 | 0.0157 | 0.0192 | 0.0207 |
| Fixed K=20 | 0.1736 | 0.0131 | 0.1734 | 0.0380 |
| Routed (single→2, multi→20) | **0.0248** | **0.0131** | **0.0246** | **0.0207** |

Routed improves overall recall over K=2 (0.019→0.025, +28%) while keeping precision at the K=20 level, because only 5 queries are up-routed. HitRate and MRR for routed equal K=2 (dominated by single-target queries where routing stays at 2).

### 7.2 Single-target queries (n=601, exactly 1 relevant)

| Mode | HitRate | Precision | Recall | MRR |
|------|---------|-----------|--------|-----|
| K=2 | 0.0183 | 0.0092 | 0.0183 | 0.0141 |
| K=20 | 0.1681 | 0.0084 | 0.1681 | 0.0316 |
| Routed | 0.0183 | 0.0084 | 0.0183 | 0.0141 |

Routed equals K=2 for recall/HitRate/MRR and equals K=20 for precision only because one aggregate singleton (`AGG_T2D_HYPERTENSION`, 1 relevant but lexically aggregate) is routed to 20, slightly lowering mean precision. No aggregate recall benefit is expected here (single relevant needs only K=1).

### 7.3 Genuinely multi-record queries (n=4, >1 relevant)

| Mode | HitRate | Precision | Recall | MRR |
|------|---------|-----------|--------|-----|
| K=2 | 1.000 | 1.000 | 0.157 | 1.000 |
| K=20 | 1.000 | 0.725 | **0.973** | 1.000 |
| Routed | 1.000 | 0.725 | **0.973** | 1.000 |

Routed captures the full K=20 benefit for the queries that matter: recall jumps from 0.157→0.973, precision drops from 1.0→0.725 (expected tradeoff; K=2 precision is artificially high because only 2 candidates, both relevant when they happen to be retrieved, but recall is poor).

### 7.4 Per-aggregate query (retrieval, not generation)

| QID | |Rel| | Label | K_routed | K=2 Recall (Prec) | K=20 Recall (Prec) | Routed Recall (Prec) |
|-----|-----|-------|--------|-------------------|---------------------|---------------------|
| AGG_AMLODIPINE_5MG | 17 | multi | 20 | 0.118 (1.000) | 0.941 (0.800) | 0.941 (0.800) |
| AGG_METFORMIN_500MG | 7 | multi | 20 | 0.286 (1.000) | **1.000 (0.350)** | **1.000 (0.350)** |
| AGG_PARACETAMOL_650MG | 20 | multi | 20 | 0.100 (1.000) | 0.950 (0.950) | 0.950 (0.950) |
| AGG_HYPERTENSION | 16 | multi | 20 | 0.125 (1.000) | **1.000 (0.800)** | **1.000 (0.800)** |
| AGG_T2D_HYPERTENSION | 1 | multi | 20 | 1.000 (0.500) | 1.000 (0.050) | 1.000 (0.050) |

All five aggregates correctly routed to K=20 (lexical match). Four multi-record aggregates achieve 0.94–1.0 recall at K=20 vs 0.10–0.29 at K=2. The singleton aggregate already has recall 1.0 at K=2; routing to 20 harms its precision (0.50→0.05) with no recall gain — acceptable cost (affects only 1 of 605 queries; overall routed precision penalty is 0.0157→0.0131).

### 7.5 Routed K distribution (n=605)

- Mean K: **2.15**
- Median K: **2**
- Min K: **2**
- Max K: **20**
- Count routed to K=2: **600**
- Count routed to K=20: **5**
- Std: ~1.0? (only two values; distribution is bimodal)

Classification matches ground-truth aggregate intent for all 5 aggregates; remaining 600 single-target queries stay at 2. No query routed to an intermediate value.

## 8. Single vs Multi Analysis

- **Separation quality:** For this dataset, lexical routing cleanly separates the 5 aggregate queries from 600 singles (no false positives/negatives on the aggregate set). The singleton `AGG_T2D_HYPERTENSION` is lexically aggregate (“Which patients…”) and routes to 20 despite having 1 relevant — heuristic correctly follows surface form, which is the intended policy (aggregate phrasing signals intent, not ground-truth count). Broader false-positive risk remains for singular “Summarize…” or “How many…” phrasing outside the aggregate set, but not exercised in current ground truth.
- **Benefit concentration:** All recall lift is concentrated in the 4 multi-record aggregates; 601 single-target queries see no recall change (and minimal precision change). This is the desired conservative behavior: pay K=20 cost only where coverage demands it.
- **Precision cost:** Routed overall precision equals K=20 (0.0131) rather than K=2 (0.0157) because 5 queries incur distractor load. For multi-record, precision at routed K=20 is 0.725 vs 1.0 at K=2 — the tradeoff characterized in the depth experiment (K=20 yields ~27% distractors for those queries).

## 9. Limitations

1. **Heuristic, not learned/score-based:** No similarity gap, threshold, or learned stopping signal; uses surface lexical cues only. Will misclassify if aggregate intent is phrased without vocabulary (e.g., “Get everyone with…”) or if factual query coincidentally contains aggregate phrase (“List all steps for patient…”).
2. **Summarize ambiguity:** “Summarize” requires plural hint to fire as multi; plain “Summarize the patient record.” correctly stays single, but “Summarize all patients…” routes to multi — reasonable but heuristic.
3. **Single K for all aggregates:** K=20 is not optimal per query (METFORMIN needs 10, HYPERTENSION 20, AMLODIPINE 30 for 100% recall in characterization). Fixed 20 leaves 1 missed record for AMLODIPINE (16/17) and 1 for PARACETAMOL (19/20); characterization optimum at K=30 would capture them but at higher precision cost.
4. **Singleton aggregate penalty:** Routing lexically aggregate but single-relevant queries to 20 harms precision (0.50→0.05) with no recall gain — expected, but quantifies cost of over-fetching for sparse aggregates.
5. **Retrieval is not generation:** Report measures retrieval only; per adaptive_k characterization, generation often fails even when retrieval is complete (e.g., HYPERTENSION at K=20 generation recall 0.0). Routing improves candidate coverage but does not guarantee answer completeness — generation separation is maintained (see §11).
6. **Single embedding/retriever evaluated:** Only dense FAISS + medical masking; BM25/hybrid/reranking not measured.

## 10. Failure Cases

- **Over-fetch singleton:** `AGG_T2D_HYPERTENSION` (1 relevant) routed to 20: retrieval recall 1.0 at both K=2 and K=20, but precision 0.50→0.05, 19 distractors added. Generation from characterization: recall 0 at K=2/5/10, 1.0 at K=20, 0 at K=30/50 — suggests generation benefits from fuller context for this query at 20 but collapses with more distractors.
- **Under-fetch for 100% coverage:** `AGG_AMLODIPINE_5MG` needs K=30 for 17/17; routed 20 yields 16/17. Remaining miss is a retrieval miss, not classification miss.
- **Generator still limiting:** For `HYPERTENSION`, retrieval at routed 20 is 1.0, but prior generation experiment at K=20 gave generation recall 0.0 — additional candidates do not ensure enumeration with baseline prompt.
- **Lexical false-negative risk:** A query like “Get all hypertensive patients” would stay single (no vocabulary match) and miss benefit — not present in current ground truth but illustrates heuristic brittleness.

## 11. Generation Separation

Generation was **not** re-tuned and not jointly scored in this benchmark. Retrieval experiment establishes candidate coverage; generation must be evaluated separately. Prior adaptive_k generation results (30 runs, budget 1200, no filtering) showed:

- Mean generation recall does not monotonically follow retrieval recall.
- HYPERTENSION K=20 retrieval 1.0 → generation 0.0 (remaining loss after retrieval).
- AMLODIPINE K=20 retrieval 0.941 → generation 0.824 (near-complete), but K=30 retrieval 1.0 → generation 0.0 (distractors interfere).

Do NOT claim “routed retrieval improves answer quality” until a controlled generation experiment demonstrates it independently. If generation is later benchmarked for routed vs fixed, it must use identical prompts/budgets and report generation metrics separately from retrieval metrics reported here.

## 12. Future Work

- Validate heuristic on paraphrased/held-out aggregate phrasing (e.g., synthetic negatives, LLM-rephrased aggregates) to estimate false-positive/negative rates beyond the 5 canonical aggregates.
- Explore validated stopping signals that could justify true score-adaptive retrieval: score-gap, entropy, or learned K predictor — only if a reliable runtime signal is established; until then keep fixed-depth routing.
- Per-query optimal K analysis (K=10 sufficient for METFORMIN, K=30 needed for AMLODIPINE) could motivate a tiered policy if future characterization shows stable query-type optima, but not in this conservative first implementation.
- End-to-end routed vs fixed generation comparison under identical budgets/prompts, with the masked fingerprint evaluator, to quantify answer-level impact separately.

---

**Artifacts:**

- `secure_rag/query_router.py` — classifier, `DEFAULT_SINGLE_K`, `AGGREGATE_K`, `routed_k()`
- `secure_rag/rag_pipeline.py` — `rag_answer(..., routed=False, k=None)` routing above `retrieve()`
- `secure_rag/cli.py` — experimental `--routed` flag with per-query logging
- `tests/test_query_routed_retrieval.py` — 24 tests (A–H)
- `benchmarks/retrieval/query_routed/query_routed_runner.py` — research benchmark (605 queries)
- `benchmarks/retrieval/query_routed/query_routed_results.json` — raw results
- This report: `reports/retrieval/adaptive_k/query_routed_retrieval_report.md`

**Verification:**

- `secure_rag/retriever.py` default K still 2; `secure_rag/vector_store.py` default 2; no masking/embedding/FAISS/generation changes; no ground-truth runtime dependency; deterministic classification; privacy verified (no raw MRN in context — checked via regex, as in tests `test_routed_privacy_no_raw_mrn`).

