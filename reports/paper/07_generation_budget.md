# 07 — Generation Budget (v2, Valid)

**Original report:** `reports/retrieval/generation/generation_budget_report_v2.md` (supersedes `generation_budget_report.md` which is HISTORICAL/INVALID)

**Artifacts:** `benchmarks/retrieval/generation/generation_results_v2.json` (1.38M, 25 runs), `generation_metrics_v2.json`, `generation_metrics.py`, `generation_budget_runner.py`; instrumentation in `secure_rag/generator.py` (`get_last_ollama_metadata`, `done`/`done_reason`/`eval_count`)

> **Historical note:** v1 measured MRN recall via regex `MRN\d+` on output while context contained zero raw MRNs (`[PATIENT_ID_MASKED]`), guaranteeing 0/16; plus heuristic truncation. v1 is preserved as `generation_results_v1.json` + `generation_budget_report.md` but labeled invalid. Paper MUST use v2.

## 1. Objective
Test whether generation output budget limits aggregate enumeration when retrieval is frozen complete (K=20, 16/16 hypertension records).

## 2. Dataset / Query Population
- Query: `Give me all patients with hypertension.` (AGG_HYPERTENSION, 16 relevant). Dataset 120 records. Retrieval frozen at K=20 (16/16, order frozen, context hash 93eb5f…, zero raw MRNs, masked placeholder present).

## 3. Method
- **Provider:** Ollama `llama3.2:latest`, temperature 0.3, `LLM_PROVIDER=ollama`, `num_predict` = budget, `stream=True`, provider metadata captured via `get_last_ollama_metadata`.
- **Context:** 20 masked chunks (16 relevant +4 distractors CKD/T2D), identical every run, 3415 prompt tokens (`prompt_eval_count`).
- **Measurand (v2 corrected):** Enumerated patient/record coverage via deterministic masked-record fingerprint matching (no raw MRN required). Score per enumerated segment vs 20 fingerprints (age_num +10 mandatory, age+gender +2, diagnosis +5, secondary +5, treatment +2, admission +2, threshold ≥12). Recall = distinct GT matched/16. Also missing, duplicate, unsupported (non-relevant matched + hallucinated), enumerated count, truncation via `done_reason`.
- **Budgets:** 200/400/600/800/1200 ×5 reps =25 runs; pilot 2-run mandatory before full.

## 4. Experimental Controls
- Retrieval/context/model/temperature/masking/ground-truth/K frozen; only `num_predict` varies.

## 5. Metrics
- Per-run: recall, found/missing/dup/unsupported/enum/halluc, output_tokens (`eval_count`), prompt_tokens, finish_reason, truncation_status, latency, answer_text/prompt/metadata.
- Aggregate: mean recall, std, min–max, means for missing/dup/unsupported, truncation_rate, mean tokens/latency.

## 6. Main Numerical Results

Aggregate by budget (from `generation_metrics_v2.json`):

| budget | mean_recall | std | min–max | mean_missing | mean_dup | mean_unsupported | trunc_rate | mean_out_tokens | mean_latency_ms |
|--------|-------------|-----|---------|--------------|----------|------------------|------------|-----------------|-----------------|
| 200 | 0.175 |0.025|0.125–0.188|13.2|0.4|0.8|1.00 (5/5)|200|22444|
| 400 | 0.412 |0.122|0.312–0.562|9.4|0.4|0.4|1.00|400|30474|
| 600 | 0.562 |0.125|0.500–0.812|7.0|0.2|0.6|1.00|600|48522|
| 800 | 0.737 |0.092|0.625–0.812|4.2|0.8|3.2|0.40 (2/5)|743.8|77443|
| 1200| 0.825 |0.025|0.812–0.875|2.8|0.2|3.6|0.60 (3/5)|1084|98851|

Monotonic increase 0.175→0.825 but saturates below 1.0; max 14/16 (0.875) at 1200 rep4; no 16/16.

Per-budget notes: 200–600 all `length` truncated (100%); 800 3/5 `stop` at 705–707 tokens recall 0.812, 2/5 `length` at 800 recall 0.625; 1200 3/5 `length` at 1200 recall 0.812, 2/5 `stop` at 707/1113 recall 0.812–0.875. Unknown terminations 0. Provider tokens estimated false, prompt 3415 always.

## 7. Interpretation
Budget is significant bottleneck ≤600 (all truncated, recall ≤0.56). At 800–1200 partially bottleneck but recall saturates ~0.825, remaining loss not solely budget (some `stop` runs still miss 3). Unsupported enumeration grows 0.8→3.6 with budget (model enumerates distractors). Hypothesis partially supported.

## 8. Limitations
Single query/dataset, temperature 0.3, llama3.2, 5 reps, fingerprint proxy (age+diagnosis), no human adjudication.

## 9. Artifact / Reproduction Path
- `python3 -m benchmarks.retrieval.generation.generation_budget_runner` (+ `--pilot-only`)
- Runners/metrics/generator instrumentation preserved.
