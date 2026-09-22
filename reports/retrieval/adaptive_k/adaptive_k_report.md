# Adaptive Retrieval Depth Experiment

## 1. Research Question

"What retrieval depth provides an appropriate tradeoff between retrieval recall, retrieval precision, and end-to-end generation coverage?"

We evaluate retrieval and generation separately to distinguish:
- retrieval failure (relevant records not in top-K)
- context failure (relevant not included in LLM context)
- generation failure (relevant in context but not enumerated in answer)

## 2. Motivation

Previous experiments established:

- K=2 is insufficient for many aggregate queries.
- At K=20, the hypertension query retrieves all 16 relevant records.
- Context construction preserves all retrieved records.
- Generation can still fail to enumerate all relevant records even when retrieval is complete.
- Increasing K beyond the point of full retrieval introduces additional distractors.
- Generation filtering reduces unsupported enumeration but reduces recall under fixed budgets.

The question now is to characterize how retrieval depth K affects retrieval quality, multi-record coverage, and end-to-end generation coverage without conflating stages.

## 3. Experimental Setup

- **K values evaluated exactly:** 2, 5, 10, 20, 30, 50
- **Queries:** 5 aggregate ground-truth queries (existing `ground_truth_v2.json`, not regenerated, not mutated)
  - AGG_HYPERTENSION (16 relevant)
  - AGG_AMLODIPINE_5MG (17 relevant)
  - AGG_PARACETAMOL_650MG (20 relevant)
  - AGG_METFORMIN_500MG (7 relevant)
  - AGG_T2D_HYPERTENSION (1 relevant)
- **Retrieval:** existing Dense FAISS, existing embedding model `all-MiniLM-L6-v2`, existing medical masking (`medical` policy + detector stack), existing `VectorStore`, existing retrieval implementation. K passed explicitly from research runner; production default K=2 unchanged.
- **Ground truth verified programmatically before running** — counts match specification.
- **Index:** single masked index (medical pre-embedding masking), 120 records → 120 chunks (1 chunk/record), deterministic, same query processing for all K.
- **Generation:** Ollama `llama3.2:latest`, temperature 0.3, generation budget 1200 tokens, baseline v2 prompt (no filtering instruction), 30 runs total (5 queries × 6 K values, one repetition each).
- **Privacy:** raw MRNs never enter prompt/context; verified via regex for `\bMRN\d+\b`; masked placeholder `[PATIENT_ID_MASKED]` present.
- **Metrics:** retrieval — HitRate, Precision, Recall, MRR per benchmark definitions (`benchmarks/retrieval/metrics.py`); generation — corrected v2 masked-record fingerprint evaluator (`benchmarks/retrieval/generation/generation_metrics.py`).

Artifacts are research-only under `benchmarks/retrieval/adaptive_k/` and `reports/retrieval/adaptive_k/`. Production prompt, masking, embeddings, dataset, ground truth remain unchanged.

## 4. Queries and Ground Truth

Ground truth `benchmarks/retrieval/ground_truth_v2.json` validated programmatically:

| QID | Question | Relevant Records | Category |
|-----|----------|------------------|----------|
| AGG_HYPERTENSION | Which patients have Hypertension? | 16 | multi_record_retrieval |
| AGG_AMLODIPINE_5MG | Which patients were prescribed Amlodipine 5mg? | 17 | multi_record_retrieval |
| AGG_PARACETAMOL_650MG | Who received Paracetamol 650mg? | 20 | multi_record_retrieval |
| AGG_METFORMIN_500MG | Which patients were prescribed Metformin 500mg? | 7 | multi_record_retrieval |
| AGG_T2D_HYPERTENSION | Which patients are being treated for both Type 2 Diabetes and Hypertension? | 1 | multi_record_retrieval |

Statistics: total_queries 605 in full file; adaptive-K uses 5 aggregate queries. No ground-truth mutation; verification via `AGGREGATE_QUERIES_V2` constant matches file.

## 5. Retrieval Method

- Built masked index once via `mask_text(..., policy=medical, detectors=medical)` → `chunk_text` → `embed_chunks` → `VectorStore`.
- For each query/K, called `embed_chunks([query])` → `vector_store.search(k=K)` with explicit K.
- Retrieved chunk indices, record IDs, distances/scores recorded; deduplicated relevant count computed.
- All 120 records retrieved deterministically; `VectorStore` caps `K > ntotal` via `min(k, ntotal)` (not exercised because K=50 < 120).
- Retrieved details include rank, chunk_index, record_id, score, relevant flag (ground-truth membership).

Raw MRN counts in retrieved chunks: 0 (masked). Scores are L2 distances from FAISS.

## 6. Retrieval Metrics

Definitions reused from `benchmarks/retrieval/metrics.py`:

- **HitRate@K:** 1 if at least one relevant retrieved, else 0 (0 if relevant empty)
- **Precision@K:** |relevant ∩ retrieved| / K (deduplicated unique records)
- **Recall@K:** |relevant ∩ retrieved| / |relevant| (0 if no relevant)
- **MRR@K:** 1/rank of first relevant (unique record order, 0 if none)

All metrics deduplicate `record_id` via set.

Aggregated by K across:
- A. all aggregate queries (5)
- B. multi-record queries (4: HYP 16, AMLO 17, PARA 20, MET 7)
- C. single-target aggregate query (T2D+HYPERTENSION, 1)

## 7. Retrieval Results

### Per-Query Retrieval Recall

| Query | K=2 | K=5 | K=10 | K=20 | K=30 | K=50 |
|-------|-----|-----|------|------|------|------|
| AGG_HYPERTENSION (16) | 0.125 (2) | 0.312 (5) | 0.625 (10) | **1.000** (16) | 1.000 (16) | 1.000 (16) |
| AGG_AMLODIPINE_5MG (17) | 0.118 (2) | 0.294 (5) | 0.588 (10) | 0.941 (16) | **1.000** (17) | 1.000 (17) |
| AGG_PARACETAMOL_650MG (20) | 0.100 (2) | 0.250 (5) | 0.500 (10) | 0.950 (19) | **1.000** (20) | 1.000 (20) |
| AGG_METFORMIN_500MG (7) | 0.286 (2) | 0.714 (5) | **1.000** (7) | 1.000 (7) | 1.000 (7) | 1.000 (7) |
| AGG_T2D_HYPERTENSION (1) | **1.000** (1) | 1.000 (1) | 1.000 (1) | 1.000 (1) | 1.000 (1) | 1.000 (1) |

(parentheses = number relevant retrieved)

### Per-Query Precision

| Query | K=2 | K=5 | K=10 | K=20 | K=30 | K=50 |
|-------|-----|-----|------|------|------|------|
| AGG_HYPERTENSION | 1.000 | 1.000 | 1.000 | 0.800 | 0.533 | 0.320 |
| AGG_AMLODIPINE | 1.000 | 1.000 | 1.000 | 0.800 | 0.567 | 0.340 |
| AGG_PARACETAMOL | 1.000 | 1.000 | 1.000 | 0.950 | 0.667 | 0.400 |
| AGG_METFORMIN | 1.000 | 1.000 | 0.700 | 0.350 | 0.233 | 0.140 |
| AGG_T2D+HYPERTENSION | 0.500 | 0.200 | 0.100 | 0.050 | 0.033 | 0.020 |

### Aggregated Means

| K | All (5) — HitRate | Precision | Recall | MRR | Multi-record (4) — Recall | Precision | Single-target (1) — Recall | Precision |
|---|-------------------|-----------|--------|-----|---------------------------|-----------|----------------------------|-----------|
| 2 | 1.000 | 0.900 | 0.326 | 1.000 | 0.157 | 1.000 | 1.000 | 0.500 |
| 5 | 1.000 | 0.840 | 0.514 | 1.000 | 0.393 | 1.000 | 1.000 | 0.200 |
| 10 | 1.000 | 0.760 | 0.743 | 1.000 | 0.678 | 0.925 | 1.000 | 0.100 |
| 20 | 1.000 | 0.590 | **0.978** | 1.000 | 0.973 | 0.725 | 1.000 | 0.050 |
| 30 | 1.000 | 0.407 | **1.000** | 1.000 | 1.000 | 0.500 | 1.000 | 0.033 |
| 50 | 1.000 | 0.244 | **1.000** | 1.000 | 1.000 | 0.300 | 1.000 | 0.020 |

HitRate remains 1.0 across all K for this masked dense retriever — at least one relevant retrieved even at K=2 for all queries (T2D+HYPERTENSION drives this for single-target). MRR also 1.0 (first result is relevant) across all queries for this dataset/retriever.

## 8. Generation Method

- **Index/context:** same masked index as retrieval; context assembled as `"\n\n".join(retrieved_chunks)` in retrieval order, exactly K chunks, no slicing, no hidden truncation.
- **Prompt:** baseline v2 `_build_ollama_prompt(context, query)` — no filtering instruction (`Enumerate only records whose Diagnosis contains Hypertension` not added).
- **Query:** never masked; query text is original ground-truth question.
- **LLM:** Ollama `llama3.2:latest`, `LLM_PROVIDER=ollama`, temperature 0.3, `num_predict`/`max_tokens` = 1200, `options.temperature` = 0.3.
- **Runs:** 30 (5 queries × 6 K values). One repetition each. Results recorded exactly; nondeterministic variance retained.
- **Instrumentation:** Ollama streaming final metadata captured (`done`, `done_reason`, `eval_count`, `prompt_eval_count`, `total_duration` etc.); `get_last_ollama_metadata()` used. `output_tokens = eval_count` when present (`is_estimated=False`), else estimate. `prompt_tokens = prompt_eval_count` when present.
- **Truncation:** `done_reason=="length"` → `truncation=True`/`truncated`; `"stop"` → `False`/`not_truncated`; missing → `None`/`unknown`. No heuristic inference from length.
- **Privacy & integrity per run:** `context_char_count`, `context_word_count`, `prompt_tokens`, `retrieved_record_count` recorded; `raw_mrns_in_context==0` and `raw_mrns_in_prompt==0` verified, else STOP.

## 9. Generation Metrics

Reused corrected v2 masked-record fingerprint evaluator (`compute_masked_generation_metrics`):

- Builds fingerprint per retrieved masked chunk: age, age_num, diagnosis, treatment, admission reason normalized.
- Splits answer into patient segments via enumerated list markers (`\n\s*\d+\.\s*`) and age phrases.
- Scores each segment against all fingerprints: mandatory `age_num` (+10), `age+gender` (+2), diagnosis term (+5), secondary diagnosis terms (+5 each), treatment (+2), admission (+2). Best score ≥12 assigned; otherwise hallucinated sentinel.
- Computes: `relevant_records_found`, `generation_recall` (= found relevant / total relevant), `missing_records`, `duplicate_records`/`duplicate_mrns`, `unsupported_records`/`unsupported_count` (matched non-relevant retrieved + hallucinated), `enumerated_patient_count`, `hallucinated_count`, `segment_count`, `found_mrns_ordered`, `found_gt_ordered`, `output_tokens`, `prompt_tokens`, `finish_reason`, `truncation`, `latency_ms`.

No raw-MRN extraction; no second incompatible evaluator.

## 10. Generation Results

| Query | K | Retrieval Recall | Generation Recall | Relevant Found | Enumerated | Unsupported | Output Tokens | Latency ms | Truncation |
|-------|---|------------------|-------------------|----------------|------------|-------------|---------------|------------|------------|
| HYPERTENSION | 2 | 0.125 | 0.000 | 0/16 | 0 | 0 | 36 | — | not_truncated |
| HYPERTENSION | 5 | 0.312 | 0.000 | 0/16 | 5 | 5 | 79 | — | not_truncated |
| HYPERTENSION | 10 | 0.625 | 0.000 | 0/16 | 0 | 0 | 13 | — | not_truncated |
| HYPERTENSION | 20 | 1.000 | 0.000 | 0/16 | 11 | 11 | 179 | — | not_truncated |
| HYPERTENSION | 30 | 1.000 | 0.062 | 1/16 | 8 | 6 | 253 | — | not_truncated |
| HYPERTENSION | 50 | 1.000 | 0.000 | 0/16 | 2 | 2 | 91 | — | not_truncated |
| AMLODIPINE | 2 | 0.118 | 0.000 | 0/17 | 0 | 0 | 30 | — | not_truncated |
| AMLODIPINE | 5 | 0.294 | 0.059 | 1/17 | 5 | 4 | 112 | — | not_truncated |
| AMLODIPINE | 10 | 0.588 | 0.000 | 0/17 | 10 | 10 | 165 | — | not_truncated |
| AMLODIPINE | 20 | 0.941 | **0.824** | **14/17** | 16 | 2 | 573 | — | not_truncated |
| AMLODIPINE | 30 | 1.000 | 0.000 | 0/17 | 0 | 0 | 163 | — | not_truncated |
| AMLODIPINE | 50 | 1.000 | 0.000 | 0/17 | 0 | 0 | 79 | — | not_truncated |
| PARACETAMOL | 2 | 0.100 | 0.000 | 0/20 | 1 | 1 | 30 | — | not_truncated |
| PARACETAMOL | 5 | 0.250 | 0.200 | 4/20 | 4 | 0 | 96 | — | not_truncated |
| PARACETAMOL | 10 | 0.500 | 0.000 | 0/20 | 7 | 7 | 124 | — | not_truncated |
| PARACETAMOL | 20 | 0.950 | 0.000 | 0/20 | 0 | 0 | 175 | — | not_truncated |
| PARACETAMOL | 30 | 1.000 | 0.050 | 1/20 | 6 | 5 | 137 | — | not_truncated |
| PARACETAMOL | 50 | 1.000 | 0.000 | 0/20 | 1 | 1 | 163 | — | not_truncated |
| METFORMIN | 2 | 0.286 | 0.143 | 1/7 | 1 | 0 | 36 | — | not_truncated |
| METFORMIN | 5 | 0.714 | 0.000 | 0/7 | 5 | 5 | 87 | — | not_truncated |
| METFORMIN | 10 | 1.000 | 0.000 | 0/7 | 7 | 7 | 118 | — | not_truncated |
| METFORMIN | 20 | 1.000 | 0.000 | 0/7 | 10 | 10 | 162 | — | not_truncated |
| METFORMIN | 30 | 1.000 | 0.000 | 0/7 | 4 | 4 | 128 | — | not_truncated |
| METFORMIN | 50 | 1.000 | 0.000 | 0/7 | 0 | 0 | 100 | — | not_truncated |
| T2D+HYP | 2 | 1.000 | 0.000 | 0/1 | 0 | 0 | 46 | — | not_truncated |
| T2D+HYP | 5 | 1.000 | 0.000 | 0/1 | 4 | 4 | 124 | — | not_truncated |
| T2D+HYP | 10 | 1.000 | 0.000 | 0/1 | 5 | 5 | 101 | — | not_truncated |
| T2D+HYP | 20 | 1.000 | **1.000** | **1/1** | 10 | 6 | 282 | — | not_truncated |
| T2D+HYP | 30 | 1.000 | 0.000 | 0/1 | 1 | 1 | 518 | — | not_truncated |
| T2D+HYP | 50 | 1.000 | 0.000 | 0/1 | 0 | 0 | 253 | — | not_truncated |

Notes:
- At K=20, AMLODIPINE achieves 0.824 recall (14/17) and T2D+HYP achieves 1.0 (1/1) — the only cases where generation enumerates most retrieved relevant records.
- All other query/K combinations show generation recall 0.0–0.2 despite retrieval recall up to 1.0, indicating generation-stage failure dominates for this prompt/model/temperature.
- No runs were truncated (`done_reason` consistently `stop`, not `length`) at budget 1200 because answers were short (13–573 tokens) — truncation not a limiting factor here.
- Latencies varied per run; output tokens well below 1200 budget.

## 11. Retrieval Recall vs Generation Recall

Critical separation — for every query:

| Query | K | Retrieval Recall | Generation Recall | Interpretation |
|------|---|------------------|-------------------|---------------|
| HYPERTENSION | 2 | 0.125 | 0.000 | Retrieval incomplete; generation cannot be interpreted independently |
| HYPERTENSION | 5 | 0.312 | 0.000 | Retrieval incomplete |
| HYPERTENSION | 10 | 0.625 | 0.000 | Retrieval incomplete |
| HYPERTENSION | 20 | 1.000 | 0.000 | **Remaining loss occurs after retrieval** |
| HYPERTENSION | 30 | 1.000 | 0.062 | Remaining loss after retrieval; larger K did not improve generation |
| HYPERTENSION | 50 | 1.000 | 0.000 | Remaining loss after retrieval |
| AMLODIPINE | 2 | 0.118 | 0.000 | Retrieval incomplete |
| AMLODIPINE | 5 | 0.294 | 0.059 | Retrieval incomplete |
| AMLODIPINE | 10 | 0.588 | 0.000 | Retrieval incomplete |
| AMLODIPINE | 20 | 0.941 | 0.824 | Near-complete retrieval; generation tracks retrieval but still misses 3/17 |
| AMLODIPINE | 30 | 1.000 | 0.000 | Retrieval complete but generation collapsed — distractors or nondeterminism dominate |
| AMLODIPINE | 50 | 1.000 | 0.000 | Retrieval complete but generation collapsed |
| PARACETAMOL | 2 | 0.100 | 0.000 | Retrieval incomplete |
| PARACETAMOL | 5 | 0.250 | 0.200 | Retrieval incomplete but generation matches 4/5 retrieved |
| PARACETAMOL | 10 | 0.500 | 0.000 | Retrieval incomplete |
| PARACETAMOL | 20 | 0.950 | 0.000 | Retrieval near-complete but generation fails entirely |
| PARACETAMOL | 30 | 1.000 | 0.050 | Retrieval complete but generation still fails |
| PARACETAMOL | 50 | 1.000 | 0.000 | Retrieval complete but generation fails |
| METFORMIN | 2 | 0.286 | 0.143 | Retrieval incomplete |
| METFORMIN | 5 | 0.714 | 0.000 | Retrieval incomplete |
| METFORMIN | 10 | 1.000 | 0.000 | **Remaining loss occurs after retrieval** |
| METFORMIN | 20 | 1.000 | 0.000 | Remaining loss after retrieval |
| METFORMIN | 30 | 1.000 | 0.000 | Remaining loss after retrieval |
| METFORMIN | 50 | 1.000 | 0.000 | Remaining loss after retrieval |
| T2D+HYP | 2 | 1.000 | 0.000 | Remaining loss after retrieval even though retrieval fully satisfied |
| T2D+HYP | 5 | 1.000 | 0.000 | Remaining loss after retrieval |
| T2D+HYP | 10 | 1.000 | 0.000 | Remaining loss after retrieval |
| T2D+HYP | 20 | 1.000 | 1.000 | Both stages succeeded |
| T2D+HYP | 30 | 1.000 | 0.000 | Retrieval still complete but generation failed (extra distractors 29 vs 19) |
| T2D+HYP | 50 | 1.000 | 0.000 | Retrieval complete but generation failed |

If K=10 retrieval recall=0.625 and generation recall=0.0, generation cannot be interpreted independently because retrieval itself was incomplete. If K=20 retrieval recall=1.0 and generation recall=0.0, the remaining loss occurs after retrieval.

## 12. Context Growth

| K | Context char count (representative, HYPERTENSION) | Word count | Provider prompt tokens (eval_count) |
|---|---------------------------------------------------|------------|--------------------------------------|
| 2 | ~1,005 | 114 | ~366 |
| 5 | ~2,767 | 319 | ~865 |
| 10 | ~5,695 | 647 | ~1,711 |
| 20 | ~11,638 | 1,320 | ~3,422 |
| 30 | ~17,278 | 1,965 | ~4,095 (capped) |
| 50 | ~27,863 | 3,161 | ~4,095 (capped) |

Context size grows linearly with K (approx 550–600 chars per record, ~60–70 words per masked record). Prompt tokens reported by Ollama `prompt_eval_count` rise until model context limit (~4096) then saturate. Larger K increases LLM context burden proportionally to candidates while candidate coverage saturates after recall reaches 1.0.

For METFORMIN which reaches 100% recall at K=10, context at K=20 is ~2× larger without additional relevant records — purely distractors.

## 13. Precision/Recall Tradeoff

Increasing K increases retrieval recall while decreasing precision:

- **K=2:** mean precision 0.900 (multi-record 1.000), mean recall 0.326 (multi 0.157) — few distractors, but recall insufficient for multi-record coverage.
- **K=5:** precision 0.840 (multi 1.000), recall 0.514 (multi 0.393)
- **K=10:** precision 0.760 (multi 0.925), recall 0.743 (multi 0.678)
- **K=20:** precision 0.590 (multi 0.725), recall 0.978 (multi 0.973) — near-complete recall with ~27.5% distractors for multi-record.
- **K=30:** precision 0.407 (multi 0.500), recall 1.000 — full recall but 50% distractors for multi-record.
- **K=50:** precision 0.244 (multi 0.300), recall 1.000 — full recall but 70% distractors.

For single-target T2D+HYPERTENSION, precision drops from 0.500 at K=2 to 0.020 at K=50 while recall stays 1.0 — additional candidates are purely distractors and increase unsupported enumeration risk (observed: at K=20 generation enumerated 10 with 6 unsupported, at K=30 generation failed).

Larger K is not automatically better; the purpose is to characterize the tradeoff between coverage and distractor load.

## 14. Truncation and Latency

- **Generation budget:** 1200 tokens for all 30 runs; budget does not dominate K comparison (previous budget experiment showed incompleteness at 1200 even with K=20, but also showed truncation at <800).
- **Truncation:** All 30 runs reported `done_reason=="stop"` → `truncation=False` (`not_truncated`). No `length` terminations; `eval_count` well below 1200 (13–573). This contrasts with the prior budget experiment where at 1200 with the canonical hypertension query, truncation rate was 0.60 and mean `eval_count` ~1084, indicating longer enumerations. Here, answers were abbreviated, so truncation was not observed — generation incompleteness is due to model enumeration behavior, not token limit.
- **Latency:** per-run latency varied (short answers ~few seconds to ~10s+). Provider metadata includes `total_duration`, `eval_duration`, etc.; not aggregated here due to abbreviated outputs.
- **Output tokens:** ranged 13–573; prompt tokens grew with K as noted above.

## 15. Per-Query Analysis

### AGG_HYPERTENSION (16 relevant)

- Minimum K for 80% recall: **20** (recall 1.0)
- Minimum K for 90%: **20**
- Minimum K for 100%: **20**
- Precision at that K: 0.800
- Generation recall at K=20: 0.000 (retrieval complete but generation failed)
- Interpretation: Generation becomes limiting stage after retrieval reaches sufficient coverage (pattern B). Additional distractors at K=30/50 did not improve generation (recall 0.062 at K=30, 0 at K=50).

### AGG_AMLODIPINE_5MG (17 relevant)

- 80% → 20 (0.941 at 20, but 80% threshold technically satisfied at 20; 100% needs 30)
- 90% → 20
- 100% → **30** (17/17)
- Precision at K=20: 0.800, at K=30: 0.567
- Generation recall: 0.824 at K=20 (best case — additional candidate depth provided useful information), but collapsed to 0 at K=30/50 despite full retrieval (pattern C at high K). This suggests additional distractors interfere with generation when K exceeds full coverage need.

### AGG_PARACETAMOL_650MG (20 relevant)

- 80% → 20 (0.950), 90% → 20, 100% → **30**
- Precision at K=20: 0.950, at K=30: 0.667
- Generation recall: 0.200 at K=5 (4/20), 0.0 at K=10/20, 0.05 at K=30 — generation remains poor even when retrieval near-complete (pattern D: both remain poor for this query with current prompt). Limitation lies elsewhere (prompt specificity, medication ambiguity).

### AGG_METFORMIN_500MG (7 relevant)

- 80% → **10**, 90% → 10, 100% → 10 (smallest threshold among multi-record)
- Precision at K=10: 0.700 (vs 0.350 at K=20)
- Generation recall: 0.143 at K=2, 0 at K≥5 even when retrieval is 1.0 at K=10+ — strong generation bottleneck after retrieval (pattern B).

### AGG_T2D_HYPERTENSION (1 relevant)

- 80%/90%/100% → **2** (single record needs minimal K)
- Precision at K=2: 0.500, at K=50: 0.020
- Generation recall: 0 at K=2/5/10, 1.0 at K=20, 0 at K=30/50 — retrieval always 1.0, but generation only succeeded at one K value (nondeterministic). Indicates limitation lies in generation enumeration logic, not retrieval.

## 16. Cross-Query Analysis

- **Mean thresholds:** Multi-record queries with larger relevant sets (16–20) need K≈20–30 for 100% recall; smaller set (7) needs K=10. Single-target needs K=2.
- **Precision drops proportionally** to irrelevant proportion `(K - relevant_retrieved)/K`. At K=50, mean precision 0.244 means ~76% distractors.
- **Generation patterns observed:**
  - A (retrieval ↑ and generation ↑): seen only for AMLODIPINE at K=20 (and partially T2D at K=20) — additional candidates help.
  - B (retrieval ↑ but generation plateaus at 0): seen for HYPERTENSION, METFORMIN — generation is limiting stage.
  - C (retrieval ↑ but generation ↓ at high K): seen for AMLODIPINE beyond 20 and T2D beyond 20 — distractors interfere.
  - D (both poor): seen for PARACETAMOL — limitation elsewhere.
- Do not force expected outcome; multiple patterns present per spec.
- No single K eliminates tradeoff; production K=2 remains insufficient for multi-record but large K penalizes single-target precision/context.

## 17. Limitations

- Single repetition per query/K — nondeterministic LLM behavior not fully characterized (variance observed: e.g., AMLODIPINE K20 0.824 vs K30 0.0). Conclusions limited to this 30-run controlled sample.
- Masked fingerprint evaluator requires diagnosis/treatment/admission text in answer; abbreviated answers (e.g., "Patient 1: [PATIENT_ID_MASKED]") score 0 even if intent correct — may underestimate recall for terse enumerations. Reflection of prompt’s tendency to abbreviate at low K.
- No BM25/hybrid/reranking; only dense FAISS evaluated.
- No filtering instruction in primary experiment (by design); filtering experiment documented separately.
- Prompt tokens saturate at ~4095 at high K due to provider limit; actual LLM context burden higher but not reflected in `prompt_eval_count` beyond cap.
- Generation budget 1200 not varied here; prior experiment showed incompleteness even at 1200.

## 18. Conclusion

For K = 2,5,10,20,30,50 measured retrieval recall was (mean over 5 aggregate queries) 0.326, 0.514, 0.743, 0.978, 1.000, 1.000 respectively, while mean precision was 0.900, 0.840, 0.760, 0.590, 0.407, 0.244. Minimum K achieving 80%/90%/100% recall varies per query: 20/20/20 for HYPERTENSION, 20/20/30 for AMLODIPINE and PARACETAMOL, 10/10/10 for METFORMIN, 2/2/2 for T2D+HYPERTENSION. At those thresholds, precision was correspondingly reduced (e.g., 0.800 at K=20 for HYPERTENSION, 0.233 at K=30 for METFORMIN). Generation recall did not monotonically increase with retrieval recall: at K where retrieval was complete (e.g., HYPERTENSION K=20 recall 1.0, generation 0.0), remaining loss is after retrieval; at K=30/50 distractors increased without improving generation. Context size grows linearly with K (e.g., ~1k chars at K=2 to ~27k chars at K=50 for HYPERTENSION, words 114 → 3161, prompt tokens 366 → 4095). No truncation occurred at budget 1200 (all `stop`). The measured tradeoff is that larger K provides candidate coverage for multi-record queries up to K≈20–30, beyond which precision collapses and generation does not reliably benefit and may degrade.

## 19. Implications for Secure-RAG

- Production K=2 remains appropriate as default (verified unchanged: `secure_rag/retriever.py` default 2, `secure_rag/vector_store.py` default 2); it preserves precision, latency, and context size for single-record queries which dominate the dataset.
- For aggregate/multi-record use cases, retrieval at K=2 is insufficient (mean multi-record recall 0.157) — if such queries become first-class, a separate retrieval path or explicit aggregation handling is required, not a silent increase of the default K.
- No adaptive-K behavior introduced into production; K is passed explicitly from research runner only.
- Privacy preserved: raw MRNs never entered prompt/context (all runs verified 0 raw MRNs, masked placeholders present).
- Future work should investigate generation-stage enumeration (prompt specificity, structured output, or separate filtering pass) because retrieval at K=20 already achieves 97.3% mean multi-record recall, yet generation recall plateaued near 0 for most queries with the baseline v2 prompt, indicating the pipeline’s current generation limits aggregate coverage regardless of K.

---

*Artifacts:*
- `benchmarks/retrieval/adaptive_k/adaptive_k_runner.py` (produces `adaptive_k_retrieval_results.json`)
- `benchmarks/retrieval/adaptive_k/adaptive_k_metrics.py` (produces `adaptive_k_metrics.json`)
- `benchmarks/retrieval/adaptive_k/adaptive_k_generation.py` (produces `adaptive_k_generation_results.json`)
- `reports/retrieval/adaptive_k/adaptive_k_report.md` (this file)
- `tests/test_adaptive_k.py`

*Verification:*
- `secure_rag/retriever.py` default K still 2
- `secure_rag/masker.py`, `embedding.py`, `rag_pipeline.py`, `vector_store.py`, dataset, ground_truth, production prompt unchanged
- No ground-truth mutation; no BM25/Hybrid; no embedding/masking changes.

