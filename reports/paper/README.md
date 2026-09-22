# Secure-RAG Paper Research Archive — Index

Central canonical location for all paper-relevant research reports. Original detailed reports are preserved at their existing locations; this directory contains publication-facing copies/summaries.

| # | Experiment | Paper Report | Original Report | Status | Paper Relevance | Key Result |
|---|------------|--------------|-----------------|--------|-----------------|------------|
| 1 | Repository Audit | 01_repository_audit.md | reports/repository/repository_audit.md | Complete | Methods / Reproducibility | Clean baseline, production K=2 preserved |
| 2 | Dataset | 02_dataset.md | reports/dataset/validation_report.md, reports/dataset/distribution_statistics.md, data/dataset_architecture.md | Complete | Dataset description | 120 records, 33 diseases, 605 queries |
| 3 | Dense Retrieval | 03_dense_retrieval.md | reports/retrieval/dense/depth_report.md, benchmarks/retrieval/metrics.py, benchmarks/retrieval/runner.py | Complete | Baseline retrieval | Dense strong baseline; masking preserves aggregate |
| 4 | BM25 | 04_bm25.md | /RAG-bm25/reports/retrieval/bm25/bm25_report.md + depth_report.md (worktree experiment/bm25) ; summary in reports/comparison/summary.md | Complete | Comparator | BM25 competitive, no meaningful gain over Dense (k=10 recall 0.0887) |
| 5 | Hybrid | 05_hybrid.md | /RAG-hybrid/reports/retrieval/hybrid/report.md + depth_report.md (worktree experiment/hybrid) ; summary in reports/comparison/summary.md | Complete | Comparator | Hybrid ≈ Dense (k=10 recall 0.0888), adds complexity |
| 6 | Retrieval Depth | 06_retrieval_depth.md | reports/retrieval/dense/depth_report.md, benchmarks/retrieval/depth_experiment.py | Complete | Depth tradeoff | K=20≈97.8% multi recall, K=30=100% |
| 7 | Generation Budget v2 | 07_generation_budget.md | reports/retrieval/generation/generation_budget_report_v2.md | Complete | Generation bottleneck | 0.175/0.412/0.562/0.737/0.825 at 200/400/600/800/1200, saturates <1.0 |
| 8 | Generation Filtering | 08_generation_filtering.md | reports/retrieval/generation/generation_filter_report.md | Complete | Generation filtering | Recall −0.312 at 800, unsupported −3.0 (precision up, recall down) |
| 9 | Adaptive-K Characterization | 09_adaptive_k_characterization.md | reports/retrieval/adaptive_k/adaptive_k_report.md | Complete | Retrieval-gener. separation | K=20 gives 97.3% multi retrieval; generation often bottleneck remainder |
| 10 | Experimental Summary | 10_experimental_summary.md | This index + all above | Complete | Paper drafting | Consolidated defensible claims |

## Notes

- **BM25/Hybrid** primary artifacts live in worktrees `experiment/bm25` (/Users/atharvapatil/ledger/RAG-bm25) and `experiment/hybrid` (/Users/atharvapatil/ledger/RAG-hybrid). Their reports are summarized in `reports/comparison/summary.md` on master; paper copies preserve numbers verbatim.
- **Generation v1** (`reports/retrieval/generation/generation_budget_report.md` + `benchmarks/retrieval/generation/generation_results_v1.json`) is HISTORICAL/INVALID — superseded by v2 (masked-MRN measurand fix). It is preserved but NOT presented as valid; paper uses v2 only.
- **Generated JSON artifacts** (`benchmarks/retrieval/**/…/*.json`) are reproducible via runners and are intentionally ignored via `.gitignore` (except `_archive_v1`). They remain on disk for evidence but are not tracked.
- All reports in this directory are understandable without opening benchmark implementation; they list objective, dataset, method, controls, metrics, results, interpretation, limitations, reproduction path.
