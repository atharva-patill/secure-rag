# 09 — Adaptive-K Characterization

**Original report:** `reports/retrieval/adaptive_k/adaptive_k_report.md`

**Artifacts:** `benchmarks/retrieval/adaptive_k/adaptive_k_runner.py` → `adaptive_k_retrieval_results.json` (190K), `adaptive_k_metrics.py` → `adaptive_k_metrics.json` (13K), `adaptive_k_generation.py` → `adaptive_k_generation_results.json` (1.3M, 30 runs)

## 1. Objective
Characterize retrieval depth tradeoff between recall, precision, and end-to-end generation coverage, separating retrieval failure vs generation failure.

## 2. Dataset / Query Population
- 5 aggregate queries (same ground truth v2): HYPERTENSION 16, AMLODIPINE_5MG 17, PARACETAMOL_650MG 20, METFORMIN_500MG 7, T2D_HYPERTENSION 1.

## 3. Method
- **Retrieval:** Dense FAISS masked, K=2,5,10,20,30,50, explicit K passed, production default K=2 unchanged.
- **Generation:** Ollama `llama3.2:latest`, temp0.3, budget 1200, baseline v2 prompt (no filtering), 30 runs (5×6 K, 1 rep each), fingerprint evaluator, `done_reason` truncation, raw MRN check.

## 4. Experimental Controls
- Single masked index (120 records→120 chunks), deterministic, same masking/embeddings/ground-truth.

## 5. Metrics
- Retrieval: HitRate, Precision, Recall, MRR (deduplicated). Generation: fingerprint recall etc.

## 6. Main Numerical Results

Retrieval recall per query:
- HYP 16: 0.125/0.312/0.625/1.0/1.0/1.0
- AMLODIPINE 17: 0.118/0.294/0.588/0.941/1.0/1.0
- PARACETAMOL 20: 0.10/0.25/0.50/0.95/1.0/1.0
- METFORMIN 7: 0.286/0.714/1.0/1.0/1.0/1.0
- T2D_HYP 1: 1.0 at all K

Aggregated means:
- k2: HitRate1.0 Prec0.900 Rec0.326 MRR1.0 (multi recall 0.157 prec1.0)
- k5: 1.0/0.840/0.514/1.0 (multi 0.393/1.0)
- k10:1.0/0.760/0.743/1.0 (multi0.678/0.925)
- k20:1.0/0.590/0.978/1.0 (multi0.973/0.725) — K=20≈97.8% mean multi-record recall
- k30:1.0/0.407/1.0/1.0 (multi1.0/0.500) — K=30=100%
- k50:1.0/0.244/1.0/1.0 (multi1.0/0.300)

Generation recall (same table, 1200 budget, no truncation observed `stop` all 30 runs, short answers 13–573 tokens):
- HYP: 0/0/0/0/0.062/0 at k2/5/10/20/30/50
- AMLODIPINE:0/0.059/0/0.824/0/0 (peak 14/17 at k20)
- PARACETAMOL:0/0.20/0/0/0.05/0
- METFORMIN:0.143/0/0/0/0/0
- T2D_HYP:0/0/0/1.0/0/0

Interpretation table: At K where retrieval=1.0 but generation=0.0, remaining loss is after retrieval (e.g., HYP k20, METFORMIN k10+). Only AMLODIPINE k20 shows retrieval↑ and generation↑; beyond 20 distractors degrade generation.

Context growth: ~1k chars k2 →27k k50, prompt tokens 366→4095 (cap).

## 7. Interpretation
Multi-record retrieval needs K≈20–30; beyond that precision collapses and generation does not reliably benefit and may degrade. Production K=2 preserves single-record efficiency but insufficient for aggregates; separate adaptive path needed but not yet implemented. Remaining generation bottleneck dominates after retrieval complete.

## 8. Limitations
Single rep per query/K (nondeterministic LLM not fully characterized), fingerprint proxy, no BM25/hybrid, prompt tokens cap at 4095.

## 9. Artifact / Reproduction Path
- `python3 -m benchmarks.retrieval.adaptive_k.adaptive_k_runner` + `adaptive_k_generation.py`
