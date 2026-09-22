# 06 — Retrieval Depth

**Original reports:** `reports/retrieval/dense/depth_report.md` (P0 STEP 5) + `benchmarks/retrieval/depth_experiment.py`; also `reports/comparison/summary.md` §5 and `reports/retrieval/comparison/*audit.md` for context size.

## 1. Objective
Characterize recall/precision tradeoff vs K for aggregate queries.

## 2. Dataset / Query Population
- Same 120 records; focus on 5 aggregate queries (4 true multi 7–20 relevant + 1 single-target).

## 3. Method
- Dense FAISS masked, K=2,5,10,20,30,50, record-level deduplicated Recall/Precision/HitRate/MRR.

## 4. Experimental Controls
- Single masked index, frozen embeddings/masking/ground-truth.

## 5. Metrics
- Recall@K, Precision@K, coverage (smallest k for 80/90/100% recall), single vs multi means, record-level safety.

## 6. Main Numerical Results

Per-aggregate recall (baseline_a, from depth_report):
- HYPERTENSION 16: 0.125/0.312/0.625/1.0/1.0/1.0
- AMLODIPINE 17: 0.118/0.294/0.529/1.0/1.0/1.0
- PARACETAMOL 20: 0.10/0.25/0.50/0.90/0.95/1.0
- METFORMIN 7: 0.286/0.714/1.0/1.0/1.0/1.0
- T2D_HYP 1: 1.0 at all K

Coverage thresholds:
- AMLODIPINE 17: 20/20/20, METFORMIN 7: 10/10/10, PARACETAMOL 20: 20/20/50, HYP 16: 20/20/20, T2D_HYP 1: 2/2/2

Single vs multi means: multi 0.3257→0.5142→0.7309→0.9800→0.9900→1.0 from k2→50; single 0.0183→0.0433→0.0849→0.1681→0.2512→0.4176.

Precision tradeoff: at k=50 precision 0.244 mean (0.32–0.40 for 16–20 relevant) → ~60–76% distractors; context linear growth (~550 chars/record, 114→3161 words, 366→4095 prompt tokens).

Adaptive-K style aggregates (characterization experiment, §09): mean multi-record recall 0.157 at k2 → 0.393 at k5 →0.678 at k10 →0.973 at k20 →1.0 at k30 (978 vs 1.0). So K=20≈97.8% mean multi-record, K=30=100%.

## 7. Interpretation
Multi-record queries need K≈20–30 for full coverage; single-record saturates by k=10 and large K penalizes precision/context/latency. Fixed large K not optimal; adaptive candidate depth recommended but not yet implemented.

## 8. Limitations
- Only dense evaluated for this report; BM25/hybrid similar (see §04/05).

## 9. Artifact / Reproduction Path
- `benchmarks/retrieval/depth_experiment.py` → depth_report.md; adaptive-K retrieval confirms same curve.
