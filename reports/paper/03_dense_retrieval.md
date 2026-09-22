# 03 — Dense Retrieval Baseline

**Original report:** `reports/retrieval/dense/depth_report.md` (and `reports/comparison/summary.md` §3, metrics in `benchmarks/retrieval/metrics.py`, runner `benchmarks/retrieval/runner.py`)

## 1. Objective
Establish dense semantic retrieval baseline under Secure-RAG privacy preprocessing.

## 2. Dataset / Query Population
- 120 MRN records, 605 queries (same as §02). Aggregate focus: 5 aggregate queries; single-record 601 for context.

## 3. Method
- **Embeddings:** `all-MiniLM-L6-v2` via `secure_rag/embedding.py`, FAISS `IndexFlatL2`, L2 search, `min(k,ntotal)`.
- **Masking:** `medical` policy + `medical` detector stack (`RegexDetector` + `DomainDetector` + `SpaCyDetector`), pre-embedding per-record `mask_text` before `chunk_record` (1 chunk/record).
- **K evaluated:** 2,5,10,20,30,50 (depth report); runner also tests 1,3. Record-level deduplicated metrics.

## 4. Experimental Controls
- Same dataset/ground-truth/embedding/masking/vector-store across all K; no BM25/hybrid; production `secure_rag/retriever.py` default K=2 unchanged.

## 5. Metrics
- HitRate@K, Precision@K, Recall@K, MRR@K (deduplicated `record_id`); plus per-aggregate Recall@K, coverage thresholds, precision tradeoff, single vs multi comparison, record-level safety.

## 6. Main Numerical Results

Overall (605 queries, reported in `reports/comparison/summary.md` at k=10/50):
- k=10: HitRate 0.0909, Precision 0.0145, Recall 0.0888, MRR 0.0325 (masked; raw similar).
- k=50: HitRate/Recall ≈0.4215, MRR 0.0438.

Aggregate detail (from `depth_report.md` baseline_a; baseline_b identical, secure_rag similar):
- AGG_AMLODIPINE_5MG (17): k=2 0.118, k=5 0.294, k=10 0.529/0.588, k=20 1.0/0.941, k=30 1.0, k=50 1.0
- AGG_METFORMIN_500MG (7): 0.286/0.714/1.0/1.0/1.0/1.0
- AGG_PARACETAMOL_650MG (20): 0.10/0.25/0.50/0.90/0.95/1.0
- AGG_HYPERTENSION (16): 0.125/0.312/0.625/1.0/1.0/1.0
- AGG_T2D_HYPERTENSION (1): 1.0 at all K

Single vs multi means (baseline_a):
- k=2: single recall 0.0183, multi 0.3257
- k=10: single 0.0849, multi 0.7309
- k=20: single 0.1681, multi 0.9800
- k=30: single 0.2512, multi 0.9900

Masking impact at k=10 (dense): 25 degraded, 26 improved, 554 unchanged → limited per-query changes, aggregate preserved.

Record-level safety: unique MRNs, no duplicates in top-k, 1 chunk/record, passed.

## 7. Interpretation
Dense retrieval is competitive baseline; single-record poor performance is largely query-design (generic summary/PHI queries lack discriminative terms), not algorithm failure. Multi-record aggregates need K≈20–30 for full coverage; masking does not meaningfully degrade retrieval.

## 8. Limitations
- Overall HitRate dominated by 601 single-record generic queries; not representative of info-seeking lexical performance.

## 9. Artifact / Reproduction Path
- `benchmarks/retrieval/runner.py` → retrieval_results, `benchmarks/retrieval/metrics.py` → metrics, `benchmarks/retrieval/depth_experiment.py` and `failure_analysis.py` → depth/failure reports.
- `python3 -m benchmarks.retrieval.runner --version v2`
