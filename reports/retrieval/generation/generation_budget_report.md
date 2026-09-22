# Generation-Budget Experiment — Controlled Report

> **Status:** Template — results pending execution of `benchmarks/retrieval/generation/generation_budget_runner.py`.
> This report will be overwritten with measured results after the 25-generation run.

## 1. Objective

Determine whether incomplete answers for the aggregate query "Give me all patients with hypertension." (16 ground-truth records) are caused by the LLM generation output budget.

Known pre-condition: dense retrieval at K=20 already retrieves all 16 relevant records and the full 16 survive context assembly. The experiment freezes retrieval/context and varies **only** the maximum generation budget.

## 2. Hypothesis

If the generation budget is the bottleneck, larger budgets should increase record-level generation recall toward 16/16, reduce missing-record counts, and correlate with reduced truncation.

Null hypothesis: recall remains capped regardless of budget (e.g., due to prompt, model summarization behavior, or instruction-following limits).

## 3. Experimental Controls

Freeze across all 25 generations:

- query = "Give me all patients with hypertension."
- retrieved records (exact 20 at K=20, ordered)
- context ordering
- system prompt (production Secure-RAG clinical prompt)
- user prompt (`Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:`)
- model (provider-specific env: `HF_MODEL` or `OLLAMA_MODEL`)
- provider (`LLM_PROVIDER`)
- temperature = 0.3 (explicit, conservative)
- other sampling parameters (production defaults)

Only experimental variable: maximum output generation budget (`max_tokens`/`num_predict`).

## 4. Dataset / Query

- Dataset: `data/sample_patient_data.txt` — 120 medical records (deterministic, seed 42).
- Canonical query: "Give me all patients with hypertension."
- Ground truth: 16 hypertension records from `benchmarks/retrieval/ground_truth_v2.json` (qid `AGG_HYPERTENSION`):
  `MRN1001, MRN1005, MRN1021, MRN1025, MRN1026, MRN1052, MRN1059, MRN1066, MRN1070, MRN1074, MRN1085, MRN1104, MRN1107, MRN1111, MRN1118, MRN1119`.

## 5. Retrieval Verification

| Check | Expected | Actual |
|-------|----------|--------|
| K | 20 | (measured) |
| retrieved_count | 20 | (measured) |
| relevant_count | 16 | (measured) |

Verification is performed once before any generation. The experiment fails loudly if `retrieved_count != 20` or `relevant_count != 16`. The exact 20 retrieved chunks and assembled context are frozen and reused for every generation run.

## 6. Provider / Model Configuration

| Field | Value |
|-------|-------|
| provider | (measured — `LLM_PROVIDER`, default `ollama`) |
| model | (measured — `HF_MODEL` or `OLLAMA_MODEL`) |
| temperature | 0.3 |
| generation parameter | `max_tokens` (HF) / `num_predict` (Ollama) — via optional research-only parameter on `secure_rag/generator.py:generate_answer` |

Production defaults are unchanged; budget is exposed as an optional parameter only for the experiment.

## 7. Token-Budget Conditions

| Budget | Reps | Total |
|--------|------|-------|
| 200 | 5 | 5 |
| 400 | 5 | 5 |
| 600 | 5 | 5 |
| 800 | 5 | 5 |
| 1200 | 5 | 5 |

Total: 25 generations.

## 8. Results by Individual Run

> Populated after run — each row corresponds to one entry in `generation_results_v1.json`.

| run_id | budget | recall | found | missing | dup | unsupported | trunc | finish_reason | tokens | latency_ms |
|--------|--------|--------|-------|---------|-----|-------------|-------|---------------|--------|------------|
| (example) budget_200_rep_1 | 200 | 0.000 | 0 | 16 | 0 | 0 | inferred | unknown | ~est | 0 |

Raw `answer_text` for each run is stored verbatim in the JSON artifact.

## 9. Aggregate Results by Generation Budget

| budget | runs | mean_recall | std_recall | min–max | mean_missing | trunc_rate | mean_dup | mean_unsupported | mean_tokens | mean_latency_ms |
|--------|------|-------------|------------|---------|--------------|------------|----------|------------------|-------------|-----------------|
| 200 | 5 | (measured) | (measured) | (measured) | (measured) | (measured) | (measured) | (measured) | (measured) | (measured) |
| 400 | 5 | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 600 | 5 | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 800 | 5 | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 1200 | 5 | ... | ... | ... | ... | ... | ... | ... | ... | ... |

Source: `generation_metrics_v1.json` (via `benchmarks/retrieval/generation/generation_metrics.py:aggregate_by_budget`).

## 10. Mean Generation Recall

(measured per budget — see §9)

## 11. Standard Deviation

(measured per budget — see §9)

## 12. Mean Missing Records

(measured per budget)

## 13. Truncation Rate

Determined from `finish_reason` if exposed, otherwise inferred from token estimate vs budget with limitation noted. Rate = truncated_runs / 5 per budget.

## 14. Duplicate / Unsupported Record Counts

- duplicate_records: extra mentions beyond first per GT MRN.
- unsupported_records: MRNs mentioned that are not in the frozen K=20 retrieved context.

Reported as mean per budget and per-run lists.

## 15. Latency

Mean `latency_ms` per budget (wall-clock for the `generate_answer` streaming collection).

## 16. Interpretation

> To be completed only after measured results. Do NOT claim that a larger token budget solves the problem unless the data support it.

Initial (pre-run) expectation from retrieval depth analysis: K=20 already gives 100% retrieval recall for hypertension, so any remaining incompleteness is downstream. Prior end-to-end audit suggests LLM enumeration truncates.

## 17. Limitations

- Provider streaming path does not expose `finish_reason`/`usage`; truncation and token counts are inferred/estimated (see `generation_budget_runner.py:_call_generation`).
- Token estimates use heuristic tokenizer when HF tokenizer not cached.
- Single query (hypertension) — generalization to other aggregate queries not tested.
- Single temperature (0.3) — sampling variance limited to 5 reps per budget.

## 18. Conclusion

> To be completed after measured results. Summarize whether the generation budget is the bottleneck and at what budget recall saturates, if at all.

---

**Artifacts:**

- `benchmarks/retrieval/generation/generation_results_v1.json` (25 runs, exact context/prompt/answers)
- `benchmarks/retrieval/generation/generation_metrics_v1.json` (aggregates)
- This report: `reports/retrieval/generation/generation_budget_report.md`

**Exact command to reproduce:**

```bash
python3 -m benchmarks.retrieval.generation.generation_budget_runner
# or
python3 benchmarks/retrieval/generation/generation_budget_runner.py
```

**Exact provider/model/budget parameter:**

- Provider: `LLM_PROVIDER` (default `ollama`)
- Model: `HF_MODEL` (`Qwen/Qwen2.5-7B-Instruct`) or `OLLAMA_MODEL` (`llama3.2`)
- Generation parameter: `max_tokens` (HF) / `num_predict` (Ollama) — research-only optional arg on `secure_rag/generator.py:119`

**K=20 retrieval check:** must show 16/16 before generation (experiment fails otherwise).

