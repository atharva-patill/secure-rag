# 05 — Hybrid (Dense + BM25) Comparison

**Original report:** `/Users/atharvapatil/ledger/RAG-hybrid/reports/retrieval/hybrid/report.md` + `depth_report.md` + `methodology.md` (worktree `experiment/hybrid`, commit d69d3e8). Summary in `reports/comparison/summary.md`.

## 1. Objective
Test whether Hybrid Dense+BM25 via Reciprocal Rank Fusion (RRF) improves over Dense alone.

## 2. Dataset / Query Population
- Same 120 records, 605 queries, same v2 ground truth as Dense/BM25.

## 3. Method
- **Dense:** `all-MiniLM-L6-v2` FAISS `IndexFlatL2` (masked).
- **BM25:** BM25Okapi lexical (as in §04).
- **Hybrid:** RRF fusion: `score = Σ 1/(RRF_K + rank)` (RRF_K=60 default), then `sorted(-score, record_id)`. No raw Hybrid baseline (only masked). K=1,3,5,10,20,30,50.

## 4. Experimental Controls
- Same dataset/ground-truth/metrics; only retrieval fusion differs.

## 5. Metrics
- HitRate, Precision, Recall, MRR (record-level), depth.

## 6. Main Numerical Results

Hybrid masked (from summary; detailed in hybrid report):
- Overall k=10: HitRate 0.0909 Prec 0.0145 Rec 0.0888 MRR 0.0325 → **identical to Dense**; k=50: 0.4215 same as Dense/BM25.
- Per-aggregate depth: K=2 multi mean ~0.31, K=10 ~0.678, K=20 0.9728–1.0, K=30 1.0 — same as Dense.
- Hybrid report notes narrow per-aggregate edge in few cases but not meaningful overall.

Masking: raw vs masked Hybrid cannot be verified — no raw Hybrid baseline built.

Comparison to Dense at k=10 (summary table):
- Dense masked 0.0888 /0.0325, Hybrid masked 0.0888 /0.0325 → tie.
- BM25 slightly lower MRR (0.0308) but hybrid does not beat dense.

The claim “21 masking degradations” for Hybrid is **not supported** without raw Hybrid artifact — see `reports/comparison/summary.md` §10.

## 7. Interpretation
Hybrid adds complexity but provides only narrow retrieval benefit on current benchmark; Dense/Hybrid tie. Hybrid should remain research comparator, not production default based on current evidence.

## 8. Limitations
- No raw Hybrid for masking ablation; RRF params not tuned; same query-design limitation as Dense/BM25.

## 9. Artifact / Reproduction Path
- Worktree `/Users/atharvapatil/ledger/RAG-hybrid` on `experiment/hybrid`: `benchmarks/retrieval/hybrid_runner.py` → `retrieval_results_hybrid_v2.json` (6M), `metrics_hybrid_v2.json` (1M), `hybrid_depth.py` reports.
