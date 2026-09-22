# 08 — Generation Filtering

**Original report:** `reports/retrieval/generation/generation_filter_report.md`

**Artifacts:** `benchmarks/retrieval/generation/generation_filter_results.json` (10 runs, 590K), `generation_filter_metrics.json`, `generation_filter_runner.py`, `generation_filter_metrics.py`; baseline preserved `generation_results_v2.json`

## 1. Objective
Test whether explicit filtering instruction reduces unsupported enumeration and improves recall.

## 2. Dataset / Query Population
- Same as §07: hypertension 16 GT, K=20 frozen 16/16, same 120 records.

## 3. Method
- Only change: prompt Question augmented with: “Enumerate only records whose Diagnosis contains Hypertension. Do not enumerate retrieved records whose Diagnosis does not contain Hypertension…” Context byte-identical (hash 93eb5f…), zero raw MRNs, filtered prompt 3466 tokens (+51 vs v2 3415).
- Budgets tested: 800 and 1200 ×5 reps =10 runs; same fingerprint measurand; same provider/model/temperature.

## 4. Experimental Controls
- Retrieval/context/model/temperature/masking/K frozen; only instruction differs.

## 5. Metrics
- Same as §07 plus delta filtered−baseline.

## 6. Main Numerical Results

Filtered aggregates:
- 800: mean recall 0.425 std0.225 min0.312 max0.875 missing 9.2 dup 0.0 unsupported 0.2 trunc 1.00 (5/5) out 800
- 1200: 0.600 std0.200 min0.438 max0.875 missing 6.4 dup0.0 unsupported1.6 trunc0.60 (3/5) out1087.8

Baseline v2 at same budgets: 800 0.738 missing4.2 unsupported3.2 trunc0.40; 1200 0.825 missing2.8 unsupported3.6 trunc0.60.

Deltas (filtered−baseline):
- 800: Δrecall −0.312, Δmissing +5.0, Δunsupported −3.0, Δtrunc +0.60
- 1200: Δrecall −0.225, Δmissing +3.6, Δunsupported −2.0, Δtrunc 0.00

Per-run bimodal: at 800, 4/5 filtered enumerated only 5 patients verbose (recall 0.312 truncated), 1/5 14/16; at 1200, 3/5 low 0.438 truncated, 2/5 high 0.875/0.812 stop.

## 7. Interpretation
Closest to outcome B (precision up, recall not up) but more negative: unsupported decreases (−3 at 800, −2 at 1200) indicating instruction does improve selection, but recall declines and truncation rises (verbose per-patient output reduces patients-per-token). Filtering alone not sufficient; does not solve remaining missing (2–3 at 1200 best).

## 8. Limitations
Single query, two budgets, 5 reps, single instruction wording, verbosity confound, fingerprint proxy.

## 9. Artifact / Reproduction Path
- `python3 -m benchmarks.retrieval.generation.generation_filter_runner` (+ `--pilot-only`)
