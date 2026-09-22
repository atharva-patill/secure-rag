# Secure-RAG Repository Audit

## 1. Audit Date
2026-09-20T17:53Z, HEAD `ca79ebf6caf1144aa65a5db20b1237c435471482` (master, origin/master), branch `master`.

## 2. Git State
- **Current branch:** `master` (`ca79ebf` update summary report), up to date with `origin/master`.
- **HEAD:** `ca79ebf6caf1144aa65a5db20b1237c435471482`
- **Recent commits (20):** `ca79ebf` update summary report → `d8a71e6` bm25/hybrid comparison summary → `29e95de` Merge phase1 cleanup → `b2051b4` repo-structure-cleanup → `af54601` dense report → `35a069c` retrieval depth → `5bc815a` multi-record metrics → etc.
- **Branches:** `experiment/bm25` e633315, `experiment/hybrid` d69d3e8, `phase1/repo-structure-cleanup` b2051b4, `tui/animation`, `ui/TuiRefactor`.
- **Dirty before cleanup:** `M secure_rag/generator.py` (legitimate v2 Ollama instrumentation not yet committed) + 10 untracked research directories/files (`benchmarks/retrieval/adaptive_k/`, `benchmarks/retrieval/generation/`, `reports/retrieval/adaptive_k/`, `reports/retrieval/comparison/`, `reports/retrieval/generation/`, `tests/test_adaptive_k.py`, `tests/test_generation_filter_metrics.py`, `tests/test_generation_metrics.py`, `tests/test_masked_generation_metrics.py`).
- **After safe cleanup (before commit):** `M .gitignore` (extended to cover `benchmarks/retrieval/**/*.json/.md` while preserving `_archive_v1` + fixed duplicate), `M secure_rag/generator.py` (dedup `eval_duration`), `??` same untracked research files + new `?? reports/paper/` central archive (intentionally not yet committed per instructions). No reset/clean/rebase/force-push performed.

## 3. Worktrees
- `/Users/atharvapatil/ledger/RAG` → `master` ca79ebf
- `/Users/atharvapatil/ledger/RAG-bm25` → `experiment/bm25` e633315 (BM25 experiment, contains `bm25_runner.py`, `retrieval_results_bm25_v2.json` 11.7M, `metrics_bm25_v2.json` 1.9M, `depth_experiment_bm25_v2.json`, `failure_analysis_bm25_v2.json`, reports `bm25_report.md` + `depth_report.md`)
- `/Users/atharvapatil/ledger/RAG-hybrid` → `experiment/hybrid` d69d3e8 (Hybrid, contains `hybrid_runner.py`, `bm25.py`, `hybrid.py`, `retrieval_results_hybrid_v2.json` 6M, `metrics_hybrid_v2.json` 1M, reports `hybrid/report.md` etc.)
- Both worktrees contain completed work; **not deleted** per instructions.

## 4. Repository Structure

```
secure_rag/           # package (shipped) — 80 tracked files via pyproject.toml
benchmarks/           # separate evaluation code (not packaged)
  _common.py, generate_dataset.py, privacy_eval.py, README.md
  retrieval/
    _archive_v1/      # v1 legacy (tracked)
    depth_experiment.py, failure_analysis.py, ground_truth.py, metrics.py, runner.py
    ground_truth_v2.json (ignored, on disk 605 queries, 120 records)
    adaptive_k/       # untracked research — 3 py + 3 json (ignored json, but py untracked)
    generation/       # untracked research — 4 py + 5 json (ignored json)
data/
  generate_dataset.py (tracked), dataset_architecture.md (tracked), sample_patient_data.txt (ignored, 120 records)
reports/
  comparison/summary.md (tracked) — dense vs bm25 vs hybrid summary
  dataset/ (tracked) — validation_report.md, distribution_statistics.md
  retrieval/
    dense/depth_report.md (tracked)
    adaptive_k/adaptive_k_report.md (untracked)
    comparison/aggregate_runtime_audit.md, end_to_end_runtime_audit.md (untracked)
    generation/generation_budget_report_v2.md, generation_budget_report.md (historical), generation_filter_report.md (untracked)
  paper/ (new, untracked, publication-facing central archive — 10 reports + README)
  repository/repository_audit.md (this file, untracked until commit)
tests/                # 18 tracked + 4 untracked research tests
docs/, configuration files, pyproject.toml, README, .gitignore, etc.
```

`git ls-files` = 80 tracked. `git ls-files --others --ignored` shows `.env`, `__pycache__`, `benchmarks/dataset.jsonl`, `benchmarks/retrieval/ground_truth_v2.json` etc as ignored — correct.

Categories:
- **A production source:** `secure_rag/*` + `pyproject.toml` + `data/generate_dataset.py`
- **B benchmark/research:** `benchmarks/*` (adaptive_k/generation untracked but research)
- **C tests:** `tests/*` (4 new untracked)
- **D reports:** `reports/*` (paper central is new)
- **E generated artifacts:** `*.json` in benchmarks/retrieval subdirs, `*.pyc`, `__pycache__`, `.pytest_cache` — ignored or deletable
- **F archived/legacy:** `benchmarks/retrieval/_archive_v1/` (v1, tracked), `reports/retrieval/generation/generation_budget_report.md` (v1 historical)
- **G suspicious/duplicate:** duplicate `_LAST_OLLAMA_METADATA_FIELDS` entry `eval_duration` (fixed), duplicate `.gitignore` lines (fixed), top-level `__pycache__/embedding.cpython-*.pyc` (removed)
- **H should be ignored:** `benchmarks/dataset.jsonl`, `benchmarks/train_test_split.json`, `data/sample_patient_data.txt`, `__pycache__/`, `.pytest_cache/` — already in `.gitignore`; extended to `benchmarks/retrieval/**/*.json/.md`.

## 5. Production/Research Boundary

| File | Default | Status | Action |
|------|---------|--------|--------|
| `secure_rag/retriever.py:4` `k=2` | k=2 | Verified 2 | **NO CHANGE** |
| `secure_rag/vector_store.py:17` `k=2` | k=2 | Verified 2 | NO CHANGE |
| `secure_rag/masker.py` | medical pre-embedding masking | Unchanged (regex→domain+spacy two-pass) | NO CHANGE |
| `secure_rag/embedding.py` | `all-MiniLM-L6-v2` | Unchanged | NO CHANGE |
| `secure_rag/rag_pipeline.py:66-77` | `load_policy("medical")` + `load_detector_stack("medical")` per-record mask before chunking; `rag_answer` no query masking | Verified unchanged | NO CHANGE |
| `secure_rag/generator.py` | legitimate instrumentation: `get_last_ollama_metadata`, `done`, `done_reason`, `eval_count`, `prompt_eval_count`, `total_duration` etc captured without altering streaming text (`yield chunk["response"]` preserved); also research budget params `num_predict`/`max_tokens`/`temperature` optional, production defaults unchanged | **Legitimate** — preserved, only dedup fix applied | KEEP (commit after review) |

Research modifications are isolated to `benchmarks/retrieval/adaptive_k/*`, `benchmarks/retrieval/generation/*`, `tests/test_adaptive_k*` etc — no production behavior change.

## 6. Experiment Inventory

| Family | Implementation | Tests | Report | Results | Metrics | Paper Need | Status |
|--------|----------------|-------|--------|---------|---------|------------|--------|
| Dense baseline | `benchmarks/retrieval/runner.py`, `metrics.py` | `test_retrieval_metrics_v2.py`, `test_ground_truth_v2.py` | `reports/retrieval/dense/depth_report.md`, `reports/comparison/summary.md` | (via runner, K=2–50) | HitRate/Prec/Rec/MRR | Required | Active |
| Ground truth v2 | `benchmarks/retrieval/ground_truth.py` | `test_ground_truth_v2.py` | `reports/dataset/validation_report.md` | `ground_truth_v2.json` 605 queries | validation | Required | Active |
| Retrieval depth | `benchmarks/retrieval/depth_experiment.py`, `failure_analysis.py` | — | `reports/retrieval/dense/depth_report.md` | dense depth tables | Recall@K, coverage | Required | Active |
| BM25 | `experiment/bm25` worktree `bm25_runner.py` | (in worktree) | `bm25_report.md`, `depth_report.md` + summary | `retrieval_results_bm25_v2.json` 11.7M | same | Required | Active (worktree) |
| Hybrid | `experiment/hybrid` worktree `hybrid_runner.py`, `hybrid.py`, `bm25.py` | — | `hybrid/report.md` + summary | `retrieval_results_hybrid_v2.json` 6M | same | Required | Active (worktree) |
| Generation budget v1 | `generation_results_v1.json` (historical) | — | `generation_budget_report.md` (template, now historical) | invalid 0.0 MRN recall | — | **Historical/Invalid** — preserved, superseded by v2 | Legacy |
| Generation budget v2 | `benchmarks/retrieval/generation/generation_budget_runner.py`, `generation_metrics.py` | `test_generation_metrics.py`, `test_masked_generation_metrics.py` (untracked) | `generation_budget_report_v2.md` | `generation_results_v2.json` 1.38M (25 runs), `generation_metrics_v2.json` | masked fingerprint recall | Required | Active |
| Generation filtering | `generation_filter_runner.py`, `generation_filter_metrics.py` | `test_generation_filter_metrics.py` | `generation_filter_report.md` | `generation_filter_results.json` 590K, `generation_filter_metrics.json` | delta recall/unsupported | Required | Active |
| Adaptive-K characterization | `adaptive_k_runner.py`, `adaptive_k_metrics.py`, `adaptive_k_generation.py` | `test_adaptive_k.py` | `adaptive_k_report.md` | `adaptive_k_retrieval_results.json` 190K, `adaptive_k_metrics.json` 13K, `adaptive_k_generation_results.json` 1.3M (30 runs) | retrieval+generation | Required | Active |

**Do not delete:** all completed experiments preserved for reproducibility; paper needs evidence.

## 7. Legacy Infrastructure

- **v1 benchmark infra:** `benchmarks/retrieval/_archive_v1/` — 4 json (tracked) preserved, correctly ignored exception in `.gitignore`. No revert.
- **Duplicate dataset generators:** `benchmarks/generate_dataset.py` (legacy MEDxxx/Faker for `privacy_eval.py`) vs `data/generate_dataset.py` (canonical MRN 120, domain-aware) — both intentional, documented in `benchmarks/README.md`, kept separate.
- **No duplicate metric/masking:** checked, none.
- **sys.path hacks:** `grep -R sys.path.insert` → 0 in repo (venv excluded).
- **Dead imports:** only duplicate `eval_duration` in `generator.py:95-96` — fixed; no stale relative imports.
- **Generated JSON/MD ignored:** `benchmarks/retrieval/*.json/.md` + now `**/*.json/.md` (added) with `_archive_v1` exception; `data/sample_patient_data.txt` ignored.
- **Old experiment code:** none unreferenced; all active.

Cleanup previously done (phase1): v1 moved to `_archive_v1/`, dataset reports to `reports/dataset/`, depth report to `reports/retrieval/dense/`, artifacts removed from tracking — **not undone**.

## 8. Generated Artifacts

| Path | Size | Category | Action | Regenerable? |
|------|------|----------|--------|--------------|
| `benchmarks/dataset.jsonl` 146K, `dataset_queries.json` 149K, `train_test_split.json` 1.7K | generated legacy | **IGNORE** (in .gitignore) | yes `benchmarks/generate_dataset.py` | yes |
| `benchmarks/retrieval/ground_truth_v2.json` 177K, `train_test_split.json` | generated v2 ground truth | **IGNORE** (now covered via `**`) but preserved on disk | yes `ground_truth.py` | yes |
| `benchmarks/retrieval/_archive_v1/*` 6M | archived source of truth | **KEEP** (tracked, exception) | no — historical | — |
| `benchmarks/retrieval/adaptive_k/*.json` 13K–1.3M | reproducible research artifact | **IGNORE** (new `**` rule) but **PRESERVE on disk** | yes `adaptive_k_runner.py` + `adaptive_k_generation.py` | yes |
| `benchmarks/retrieval/generation/*.json` 967B–1.38M (v1 1.33M, v2 1.38M, filter 590K) | reproducible research artifact | **IGNORE** but **PRESERVE** | yes `generation_budget_runner.py` etc | yes |
| `__pycache__/`, `benchmarks/**/__pycache__/`, `data/__pycache__/`, `secure_rag/__pycache__/`, `.pytest_cache/`, `*.pyc` | caches | **DELETE** — removed | yes `compileall` | yes |
| `*.egg-info/`, `dist/`, `build/` | build | **IGNORE** already | yes `python -m build` | yes |
| `reports/paper/*.md` (10) | paper archive | **KEEP** (tracked after review) | no — hand-curated summaries | — |

No generated JSON accidentally tracked; only `_archive_v1` tracked intentionally.

## 9. Documentation Audit

- **README:** accurately describes Secure-RAG, pre-embedding masking, policy layer (`medical` domain), supported medical domain, research/benchmark structure (`benchmarks/privacy_eval.py` raw vs post vs pre), reproducibility, production API (`build_rag`, `rag_answer`). Version `0.2.0a1` in README vs `0.2.0a3` in `pyproject.toml` — minor drift but not stale. Docker/spaCy/HF_API_KEY docs correct. **No extensive rewrite** needed; only factual paths verified.
- **Stale claims searched:** `grep` for K=2 only tested, old 15/16 hypertension, old v1 MRN-recall, filtering improves recall, hybrid improves, BM25 superior — **not found** as publication claims. Reports correctly state: filtering recall decreased, hybrid ~dense, BM25 competitive not superior, K=2 insufficient but not only tested (K=2–50 tested), dataset 120 / queries 605 consistent, v1 labeled invalid.

## 10. Test Results
- `python3 -m pytest tests/ -q` → **206 passed, 2 skipped, 1 xfailed, 3 warnings** (swig, expected). Before cleanup same.
- `python3 -m compileall secure_rag benchmarks tests` → all listed directories compiled, no errors.
- `git diff --check` → **no whitespace errors** after cleanup (before and after).

## 11. Cleanup Performed

| Path | Category | Action | Reason | Risk |
|------|----------|--------|--------|------|
| `__pycache__/`, `**/__pycache__/`, `.pytest_cache/`, `*.pyc` | generated cache | **DELETE** (rm -rf) | Regenerable, noise | None |
| `secure_rag/generator.py:95-96` duplicate `"eval_duration"` | dead duplicate | **FIX** (remove second) | Clearly obsolete duplicate in `_LAST_OLLAMA_METADATA_FIELDS` | None |
| `.gitignore` | config | **DOCUMENTATION FIX** — added `benchmarks/retrieval/**/*.json` and `**/*.md` to cover subdir generated artifacts, added `!_archive_v1/**/*.json` to preserve archive recursion, deduped duplicate lines | Previously new research json in `adaptive_k/`/`generation/` were untracked not ignored (noise); now correctly ignored while py files remain untracked for review | Low — ground_truth_v2.json transitions from ignored (already) to still ignored, no tracking change |
| (not deleted) `benchmarks/retrieval/adaptive_k/`, `generation/`, `reports/retrieval/*`, `tests/test_*` | research | **KEEP** (not deleted) | Required for paper | — |
| (not deleted) `EVALUATION_FRAMEWORK_CHECKLIST.md`, `REFACTOR_CHECKLIST.md`, `rag_features_checklist.md` | ignored checklists | **KEEP** (ignored, not tracked) | Historical provenance, already ignored | Low |
| `reports/paper/` | central archive | **CREATE** (new) — 10 paper-facing reports + README | Required by Phase 7.5, original reports preserved | Low |

Destructive actions **not taken:** no `reset`, `clean -fd`, rebase, force push, branch deletion, report deletion, ground-truth/dataset/model retrieval changes, no adaptive retrieval implementation.

## 12. Files Intentionally Preserved
- All `secure_rag/*` production files (only dedup fix).
- All `benchmarks/retrieval/_archive_v1/*` legacy.
- `data/sample_patient_data.txt` (ignored, 120 records).
- `benchmarks/retrieval/ground_truth_v2.json` (ignored, 605 queries) — verified 605/120.
- All 5 dense/BM25/Hybrid depth artifacts (worktrees + summary).
- Generation v2: `generation_results_v2.json`, `generation_metrics_v2.json`, `generation_budget_report_v2.md`.
- Generation filtering: `generation_filter_runner.py`, `generation_filter_metrics.py`, `generation_filter_results.json`, `generation_filter_metrics.json`, `generation_filter_report.md`.
- Adaptive-K: `adaptive_k_runner.py`, `adaptive_k_metrics.py`, `adaptive_k_generation.py`, `adaptive_k_retrieval_results.json`, `adaptive_k_generation_results.json`, `adaptive_k_metrics.json`, `adaptive_k_report.md`.
- BM25/Hybrid worktree branches and reports (not deleted).
- Historical v1: `generation_results_v1.json`, `generation_budget_report.md` — labeled invalid.
- Original detailed reports at existing locations + new central `reports/paper/` copies.

## 13. Remaining Ambiguities
- Research code in `benchmarks/retrieval/adaptive_k/` and `generation/` (+ tests) is **untracked** — intentional per “do not commit until approved”, but will need `git add` after review. Currently py files show as `??` while json are ignored (correct).
- `generation_budget_report.md` v1 template vs `generation_budget_report_v2.md` v2: paper must use v2; v1 preserved as historical/invalid — flagged in `reports/paper/README.md` and `10_experimental_summary.md`.
- Hybrid “21 masking degradations” claim in hybrid report disagrees with `metrics_hybrid_v2.json` per `reports/comparison/summary.md` §10 — flagged, needs correction before publication but not changed in audit.
- `reports/paper/` is currently untracked; after review, should be added and `!` not needed in `.gitignore`.
- No incompatible metric numbers found across preserved reports; all numbers verbatim.

## 14. Final Repository State
- **Branch:** `master` ca79ebf, dirty `M .gitignore`, `M secure_rag/generator.py` (legitimate instrumentation + dedup), untracked research + paper archive.
- **Worktrees:** 3 (master + bm25 + hybrid) intact.
- **Tests:** 206 passed.
- **Compile:** pass.
- **diff --check:** clean.
- **Retriever default K:** 2 verified.
- **Dataset:** 120 records verified.
- **Ground truth:** 605 queries verified.
- **All completed experiment reports exist** (original + paper copies).
- **No raw sensitive dataset tracked** (sample data ignored).

## 15. Readiness for Adaptive Retrieval
**Ready:** repository is publication-ready baseline; production behavior unchanged, all evidence preserved, central paper archive created, no blocking issues. Next step is to implement adaptive retrieval on a feature branch (not on master) and evaluate against the frozen depth/budget/filtering baselines at fixed model/temperature and ≥800 tokens to avoid budget confound.

---
*Artifacts for reproduction:* `python3 -m pytest tests/ -q`, `python3 -m compileall secure_rag benchmarks tests`, `git diff --check`, `python3 data/generate_dataset.py`, `benchmarks/retrieval/ground_truth.py`, runners in `benchmarks/retrieval/*/`.
