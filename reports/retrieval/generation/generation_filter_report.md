# Generation Filtering Experiment

> **Status:** Executed 2026-09-20 — 10 runs with filtered generation instruction (research-only).
> **Artifacts:** `benchmarks/retrieval/generation/generation_filter_results.json`, `benchmarks/retrieval/generation/generation_filter_metrics.json`
> **Baseline preserved:** `generation_results_v2.json`, `generation_metrics_v2.json`, `generation_budget_report_v2.md` untouched.
> **Provider:** Ollama `llama3.2:latest`, Temperature 0.3, K=20, Dense FAISS, medical pre-embedding masking.

---

## 1. Research Question

The generation-budget v2 experiment established:

- Retrieval at K=20 contains all 16 hypertension-relevant records (frozen context contains all 16).
- Generation recall increased with budget: 200→0.175, 400→0.412, 600→0.562, 800→0.737, 1200→0.825, but never reached 16/16.
- Unsupported (non-relevant) enumeration increased at high budgets (0.8 at 200 → 3.6 at 1200).
- Some runs terminated with `done_reason="stop"` before achieving complete recall, suggesting generation-stage stopping, not just truncation.

Remaining question: **Is the remaining incompleteness / unsupported enumeration caused by the LLM failing to filter to the aggregate condition (Diagnosis contains Hypertension) during generation, and can an explicit record-selection instruction improve coverage/precision?**

This experiment isolates the generation filtering instruction while freezing retrieval, context, model, temperature, K, embeddings, masking, and ground truth.

---

## 2. Hypothesis

> "Explicitly constraining the model to enumerate ONLY records whose Diagnosis contains Hypertension may reduce unsupported enumeration and improve useful record coverage (generation recall)."

Expected outcomes defined a priori:

- A. Recall improves AND unsupported decreases → filtering was bottleneck
- B. Unsupported decreases but recall does not improve → improves precision, not recall
- C. Recall improves but unsupported also increases → changes behavior but tradeoff unclear
- D. Little/no meaningful change → instruction not explanatory

Do not force positive result; report measured deltas.

---

## 3. Experimental Controls

| Parameter | Value | Frozen? |
|-----------|-------|---------|
| Provider | Ollama | Yes |
| Model | `llama3.2:latest` (`OLLAMA_MODEL` env) | Yes |
| Temperature | 0.3 | Yes |
| Retrieval | Dense FAISS | Yes |
| K | 20 | Yes (verified 20/16) |
| Dataset | `data/sample_patient_data.txt` 120 records | Yes (120 verified) |
| Query | `Give me all patients with hypertension.` canonical | Yes |
| Ground truth | 16 hypertension MRNs (AGG_HYPERTENSION) | Yes |
| Embeddings | `embed_chunks` (sentence-transformers) | Yes |
| Masking | medical pre-embedding `mask_text` | Yes |
| Chunking | `chunk_text` | Yes |
| Ranking | FAISS order frozen | Yes |
| Context | Frozen K=20 masked chunks | Yes (hash verified) |

Production retrieval (`secure_rag/retriever.py`), masking (`secure_rag/masker.py`), embeddings (`secure_rag/embedding.py`), dataset generation, ground truth, default K=2, and production prompt defaults **unchanged**.

---

## 4. Filtering Intervention

**Baseline v2 prompt (Ollama):**

```
You are a RAG assistant. Answer only from the provided context. If the answer is not present, say 'I don't know'.

Context:
<20 masked chunks>

Question:
Give me all patients with hypertension.

Answer:
```

**Filtered prompt (ONLY change):** same context, query augmented with:

```
Enumerate only records whose Diagnosis contains Hypertension. Do not enumerate retrieved records whose Diagnosis does not contain Hypertension. Do not infer, combine, or invent patient records. Each enumerated patient must correspond to a retrieved record whose Diagnosis explicitly contains Hypertension.
```

Final filtered Question section:

```
Question:
Give me all patients with hypertension.

Enumerate only records whose Diagnosis contains Hypertension. Do not enumerate retrieved records whose Diagnosis does not contain Hypertension. Do not infer, combine, or invent patient records. Each enumerated patient must correspond to a retrieved record whose Diagnosis explicitly contains Hypertension.

Answer:
```

- Instruction is **semantic only** – no MRN list, no hidden record IDs, no evaluator info, no context modification.
- Context (20 chunks) is byte-identical to v2; only the prompt’s Question supplement changes.
- Verified per-run: `filter_instruction` field stored, `prompt` contains instruction, zero raw MRNs in prompt/context.

**Files:**

- Runner: `benchmarks/retrieval/generation/generation_filter_runner.py` (research-only, does not modify `secure_rag/generator.py` production defaults)
- Helper: `_build_filtered_ollama_prompt` and `_call_generation_filtered` normalize query to canonical + filter + Answer to guarantee single instruction placement.
- Metric: `benchmarks/retrieval/generation/generation_filter_metrics.py` re-exports v2 fingerprint scoring.

---

## 5. Dataset and Ground Truth

- **Dataset:** `data/sample_patient_data.txt` — 120 patient records, verified `len==120` at runtime. Not regenerated.
- **Ground truth (frozen):** 16 hypertension MRNs from `benchmarks/retrieval/ground_truth.py:AGG_HYPERTENSION`:

```
MRN1001, MRN1005, MRN1021, MRN1025, MRN1026, MRN1052, MRN1059, MRN1066,
MRN1070, MRN1074, MRN1085, MRN1104, MRN1107, MRN1111, MRN1118, MRN1119
```

- Sorted list used for both v2 and filtered; `GROUNDD_TRUTH_MRNS` validated against `AGGREGATE_QUERIES_V2`.
- Not modified; `ground_truth_v2.json` unchanged (605 queries, 120 records).

---

## 6. Retrieval Verification

Programmatic verification before any generation (both pilot and full runs):

- **Method:** `_load_mrn_records_raw` → `_build_masked_index` (medical masking) → `_retrieve_once` at K=20 using `embed_chunks` query vector → FAISS `VectorStore.search`.
- **Result (frozen for all 10 filtered runs and v2):**

```
Retrieved count: 20
Relevant (dedup): 16/16
Order: MRN1107, MRN1066, MRN1104, MRN1025, MRN1059, MRN1021, MRN1111, MRN1119,
       MRN1026, MRN1085, MRN1052, MRN1070, MRN1001, MRN1074, MRN1005, MRN1118,
       MRN1054, MRN1094, MRN1015, MRN1051
  16 relevant + 4 non-relevant distractors:
    MRN1054 CKD, MRN1094 CKD, MRN1015 T2D, MRN1051 T2D
Missing: [] (0)
```

- **K check:** `EXPECTED_K==20` PASS, `EXPECTED_HYPERTENSION_COUNT==16` PASS (`validate_retrieval_requirement`).
- **Record mapping:** `chunk_index → MRN` via `chunk_record_map` built outside LLM; used only for evaluation, never inserted into prompt.
- **Failure condition:** would STOP if retrieved !=20 or relevant !=16 – did not occur.

---

## 7. Context Identity Verification

- **Requirement:** filtered context byte-identical to v2 context.
- **Method:** `hashlib.sha256(context.encode()).hexdigest()` for both.
- **Values:**

```
v2_context_hash      = 93eb5f815a8fb7ef70ae3f71b1b7dbb2beea460d653c60f4ecf3c0d18619ef7a
filtered_context_hash = 93eb5f815a8fb7ef70ae3f71b1b7dbb2beea460d653c60f4ecf3c0d18619ef7a
Match: True
Context length: 11670 chars, 20 chunks
Exact filtered prompt length: 12161 chars (v2 prompt 3415 prompt tokens → filtered 3466, +51 tokens for instruction)
```

- **Verification points per run:**

```
raw_mrns_in_context: [] (0)
has_masked_placeholder: True ([PATIENT_ID_MASKED] present)
raw_mrns_in_prompt: [] (0)
has_filter_instruction: True
```

- If hash mismatched → experiment would STOP. No mismatch.
- External mapping `context_chunks[i] ↔ frozen_retrieved_ids[i]` preserved for fingerprint evaluator; raw MRNs never in LLM input.

---

## 8. Provider Configuration

- **Provider:** `LLM_PROVIDER=ollama` (env)
- **Model:** `OLLAMA_MODEL=llama3.2:latest` (env, reported per-run)
- **Temperature:** 0.30 fixed (passed via `options.temperature` and top-level `temperature` for Ollama `/api/generate`)
- **Generation budget:** `num_predict` = `max_tokens` = budget (800 or 1200) passed via `options.num_predict` and top-level `num_predict` for compatibility.
- **Streaming:** `stream=True`, Ollama `/api/generate` HTTP.
- **Instrumentation (v2 corrected, reused):** Final chunk metadata captured via `secure_rag/generator.py:_generate_ollama` global `_LAST_OLLAMA_METADATA`:

| Field | Source |
|-------|--------|
| `done` | `chunk["done"]` |
| `done_reason` | `chunk["done_reason"]` → `finish_reason` |
| `eval_count` | `chunk["eval_count"]` → `output_tokens` |
| `prompt_eval_count` | `chunk["prompt_eval_count"]` → `prompt_tokens` |
| `total_duration` | `chunk["total_duration"]` |
| `load_duration` | `chunk["load_duration"]` |
| `prompt_eval_duration` | `chunk["prompt_eval_duration"]` |
| `eval_duration` | `chunk["eval_duration"]` |
| `model` | `chunk["model"]` |
| `created_at` | `chunk["created_at"]` |

Missing fields stored as null; `truncation` derived strictly as:

```
done_reason=="length" → truncated=True
done_reason=="stop"   → truncated=False
missing/other         → truncated=None (unknown) — NOT silently False
```

All 10 filtered runs returned `done_reason` (`length` or `stop`), zero unknown, `output_tokens_is_estimated=False`, `prompt_tokens_is_estimated=False` where available.

---

## 9. Metrics

**Per-run (masked-record fingerprint, v2 measurand reused):**

1. `generation_recall` = distinct relevant GT matched /16
2. `relevant_records_found` (0–16)
3. `missing_records` / `missing_count`
4. `duplicate_records` / `duplicate_mrns` (extra mentions beyond first)
5. `unsupported_count` / `unsupported_records` (matched non-relevant + hallucinated age-present segments without fingerprint score ≥12)
6. `enumerated_patient_count` (patient-like segments, age phrase present)
7. `output_tokens` (`eval_count` provider)
8. `prompt_tokens` (`prompt_eval_count`)
9. `latency_ms` wall-clock
10. `finish_reason` (`done_reason` exact)
11. `truncation` / `truncation_status`

**Scoring (deterministic, auditable, reuse v2):**

```
+10 age_num (\d+-year-old) mandatory
+2  age+gender
+5  first diagnosis term (hypertension)
+3  full diagnosis snippet 40 chars
+5  secondary diagnosis term (e.g., iron deficiency anaemia distinguishes MRN1104 vs MRN1107)
+2  first treatment drug
+2  admission reason phrase
Threshold >=12 → assigned
```

**Aggregate by budget (via `generation_metrics.py:aggregate_by_budget`):**

- mean recall, std recall, min/max, mean missing, mean duplicates, mean unsupported, truncation_rate, unknown count, mean output/prompt tokens, mean latency.

Comparison computed filtered minus baseline at same budgets (800,1200).

---

## 10. Per-Run Results

Provider Ollama `llama3.2:latest`, K=20 frozen, 16/16 relevant.

### Filtered 800 (5 reps)

| run_id | recall | found | missing | dup | unsupported | enum | halluc | trunc | finish | out_tok | prompt_tok | latency_ms |
|--------|--------|-------|---------|-----|-------------|------|--------|-------|--------|---------|------------|------------|
| filter_800_rep1 | 0.312 | 5 | 11 | 0 | 0 | 5 | 0 | truncated | length | 800 | 3466 | 73831 |
| filter_800_rep2 | 0.875 | 14 | 2 | 0 | 1 | 15 | 0 | truncated | length | 800 | 3466 | 76768 |
| filter_800_rep3 | 0.312 | 5 | 11 | 0 | 0 | 5 | 0 | truncated | length | 800 | 3466 | 78411 |
| filter_800_rep4 | 0.312 | 5 | 11 | 0 | 0 | 5 | 0 | truncated | length | 800 | 3466 | 86576 |
| filter_800_rep5 | 0.312 | 5 | 11 | 0 | 0 | 5 | 0 | truncated | length | 800 | 3466 | 77499 |

- Answers 4/5 truncated at 800 with only 5 patients enumerated (verbose per-patient: first bullet MRN1104 53F-anaemia with full Treatment/Notes). One outlier rep2 enumerated 15 patients (concise bullets) reaching 14/16 even at 800 truncated.

### Filtered 1200 (5 reps)

| run_id | recall | found | missing | dup | unsupported | enum | halluc | trunc | finish | out_tok | prompt_tok | latency_ms |
|--------|--------|-------|---------|-----|-------------|------|--------|-------|--------|---------|------------|------------|
| filter_1200_rep1 | 0.438 | 7 | 9 | 0 | 1 | 8 | 0 | truncated | length | 1200 | 3466 | 102996 |
| filter_1200_rep2 | 0.875 | 14 | 2 | 0 | 4 | 18 | 0 | not_truncated | stop | 1004 | 3466 | 89393 |
| filter_1200_rep3 | 0.812 | 13 | 3 | 0 | 2 | 15 | 0 | not_truncated | stop | 835 | 3466 | 77379 |
| filter_1200_rep4 | 0.438 | 7 | 9 | 0 | 1 | 8 | 0 | truncated | length | 1200 | 3466 | 99740 |
| filter_1200_rep5 | 0.438 | 7 | 9 | 0 | 0 | 7 | 0 | truncated | length | 1200 | 3466 | 102907 |

- 3/5 truncated at 1200 (7–8 enum), 2/5 stopped early (835,1004 tokens) enumerating 15–18 patients.
- Hallucinated count 0 across all 10 runs (age present but no fingerprint ≥12 → none).
- All prompts contain filter instruction, zero raw MRNs.

**Raw artifacts:** verbatim `answer_text`, `prompt`, `ollama_metadata` per run in `generation_filter_results.json`.

---

## 11. Aggregate Results

### Filtered aggregates (from `generation_filter_metrics.json`)

| budget | runs | mean_recall | std_recall | min–max | mean_missing | mean_dup | mean_unsupported | trunc_rate | mean_out_tok | mean_prompt_tok | mean_latency |
|--------|------|-------------|------------|---------|--------------|----------|------------------|------------|--------------|-----------------|--------------|
| 800 | 5 | 0.425 | 0.225 | 0.312–0.875 | 9.2 | 0.0 | 0.2 | 1.00 (5/5) | 800.0 | 3466.0 | 78617 |
| 1200 | 5 | 0.600 | 0.200 | 0.438–0.875 | 6.4 | 0.0 | 1.6 | 0.60 (3/5) | 1087.8 | 3466.0 | 94483 |

### v2 baseline aggregates (for reference, same budgets)

| budget | mean_recall | mean_missing | mean_dup | mean_unsupported | trunc_rate | mean_out_tok |
|--------|-------------|--------------|----------|------------------|------------|--------------|
| 800 | 0.738 | 4.2 | 0.8 | 3.2 | 0.40 (2/5) | 743.8 |
| 1200 | 0.825 | 2.8 | 0.2 | 3.6 | 0.60 (3/5) | 1084.0 |

- Prompt tokens: v2 3415, filtered 3466 (+51, instruction overhead).
- Unknown terminations: 0 for both conditions (provider always returned `done_reason`).
- Latency similar: filtered 800 ~78.6s vs v2 77.4s, 1200 ~94.5s vs 98.8s.

---

## 12. Comparison Against Generation Budget v2

| Budget | Condition | Mean Recall | Missing | Duplicates | Unsupported | Truncation |
|--------|-----------|-------------|---------|------------|-------------|------------|
| 800 | v2 baseline | 0.738 | 4.2 | 0.8 | 3.2 | 0.40 |
| 800 | filtered | 0.425 | 9.2 | 0.0 | 0.2 | 1.00 |
| 1200 | v2 baseline | 0.825 | 2.8 | 0.2 | 3.6 | 0.60 |
| 1200 | filtered | 0.600 | 6.4 | 0.0 | 1.6 | 0.60 |

**Deltas (filtered – baseline):**

| Budget | Δ Recall | Δ Missing | Δ Duplicates | Δ Unsupported | Δ Truncation | Δ OutputTokens |
|--------|----------|-----------|--------------|---------------|--------------|----------------|
| 800 | **-0.312** | **+5.0** | -0.8 | **-3.0** | +0.60 | +56.2 |
| 1200 | **-0.225** | **+3.6** | -0.2 | **-2.0** | 0.00 | +3.8 |

- Negative recall delta = filtered worse; negative unsupported delta = filtered more precise (fewer distractors enumerated).
- At 800, truncation rate increased from 0.40→1.00 (all filtered 800 hit `length`, no early stops).

---

## 13. Recall Analysis

- **800 budget:** filtered mean 0.425 (6.8/16) vs v2 0.738 (11.8/16). Drop of 0.312 with large variance (std 0.225) driven by bimodal: 1/5 runs 0.875 (14/16) but 4/5 runs 0.312 (5/16). v2 had 3/5 at 0.812 and 2/5 at 0.625 (tighter, higher).
- **1200 budget:** filtered 0.600 (9.6/16) vs v2 0.825 (13.2/16). Drop 0.225, again bimodal: 2 high runs 0.875/0.812, 3 low runs 0.438. v2 1200 tightly 0.812–0.875 (std 0.025) vs filtered std 0.20.
- **Maximum recall** filtered 0.875 (14/16) equals baseline max 0.875; no filtered run reached 16/16. Baseline also never reached 16, but filtered did not improve ceiling.
- **Per-patient verbosity:** Filtered answers when low-recall (5 enum at 800) included full `Treatment: ..., Notes: ... Follow-up: ... Contact:` per bullet (verbose), whereas baseline 800 low-recall still enumerated 11 patients with concise bullets. Filtered instruction may have increased verbosity (perhaps model tries to justify Diagnosis contains Hypertension by echoing longer record excerpts), causing fewer patients to fit within same token budget and hitting `length` earlier with less coverage.

**Interpretation:** Filtering instruction does **not** improve useful record coverage on average; at these budgets it reduces mean recall and increases missing, with higher variance.

---

## 14. Unsupported Enumeration Analysis

- **Unsupported = non-relevant retrieved (MRN1054, MRN1094, MRN1015, MRN1051) enumerated + hallucinated.**
- **800:** baseline 3.2 unsupported → filtered 0.2 (-3.0). 4/5 filtered 800 runs had 0 unsupported (enumerated only 5 hypertension records before truncation), 1 run had 1. Baseline 800 had 1–5 unsupported (enumerated all 20 at high recall).
- **1200:** baseline 3.6 → filtered 1.6 (-2.0). Filtered 1200 had 0–4 unsupported vs baseline 3–5.
- **Precision effect:** Instruction does reduce unsupported enumeration, especially at 800 where truncation + filtering yields fewer distractors. However reduction comes alongside recall drop, not in isolation.
- **No hallucinated** patients in filtered (0) vs baseline 0 (both used fingerprint threshold; no age-without-match segments).
- **Duplicates:** filtered 0.0 mean vs baseline 0.8/0.2; filtered never duplicated a relevant record in these 10 runs.

**Interpretation:** Filtering instruction provides evidence of **increased precision (fewer distractors)** but at cost of recall. This matches outcome B partially (precision improves) but recall declined rather than stayed flat, so stronger negative than B.

---

## 15. Truncation Analysis

- **Classification:** `done_reason=="length"` → truncated True, `stop` → False, missing → None. No heuristics.
- **800:** filtered 5/5 `length` (1.00) vs baseline 2/5 `length` (0.40). Filtered 800 **always** exhausted budget; baseline had 3 early stops at ~705 tokens (recall 0.812) indicating model believed done. Filtered instruction appears to prevent early stopping at 800 – model continues generating more verbose bullets until budget exhausted, but with fewer patients covered.
- **1200:** both 3/5 `length`, 2/5 `stop`. Filtered `stop` runs at 835 and 1004 tokens (recall 0.812–0.875) vs baseline `stop` at 707 and 1113 tokens (0.812–0.875). Similar truncation rate, but filtered `stop` still not reaching 16/16.
- **Token accounting:** provider `eval_count` == budget when truncated (800→800, 1200→1200), <budget when stop (e.g., 835,1004). `output_tokens_is_estimated=False` for all 10. Prompt tokens 3466 filtered vs 3415 baseline (instruction adds ~51 tokens).
- **Unknown:** 0 filtered, 0 baseline.

**Interpretation:** At 800, filtering increases truncation rate (more budget-limited) and eliminates early stopping, suggesting instruction makes generation more token-hungry per record.

---

## 16. Latency Analysis

| Budget | Condition | Mean latency | Mean output tokens | Approx ms/token |
|--------|-----------|--------------|--------------------|-----------------|
| 800 | v2 | 77443 ms | 743.8 | 104 |
| 800 | filtered | 78617 ms | 800.0 | 98 |
| 1200 | v2 | 98851 ms | 1084.0 | 91 |
| 1200 | filtered | 94483 ms | 1087.8 | 87 |

- Latency scales with output tokens, not instruction alone. Filtered 800 slightly slower (+1.2s) despite same budget because all runs used full 800 tokens (vs baseline avg 743). Filtered 1200 slightly faster (-4.4s) despite similar tokens.
- Variance dominated by Ollama scheduling; not a primary metric.

---

## 17. Interpretation

**Hypothesis as stated:** filtering instruction improves recall and reduces unsupported.

**Measured outcome:** Unsupported **did decrease** (-3.0 at 800, -2.0 at 1200) and duplicates dropped to 0, but **recall also decreased** (-0.312 at 800, -0.225 at 1200) with increased variance and higher missing. At 800, truncation rate rose from 0.40→1.00.

**Which outcome is supported?**

Closest to **B** (precision improvement without recall gain) but **more negative**: recall not just flat but worse. Not A (would need recall up + unsupported down), not C (recall up + unsupported up), not D (no change). The closest neutral phrasing is:

> *Filtering improves precision/selection (fewer non-hypertension records enumerated) but does not solve the remaining recall limitation; at these budgets it reduces mean recall and increases budget-limited truncation, suggesting the instruction makes generation more verbose/token-hungry per record without helping the model enumerate more distinct relevant records within the fixed budget.*

- The filtering instruction provides evidence that generation-stage record selection **is** a controllable factor for unsupported enumeration (2–3 fewer distractors), but **is not sufficient** to improve useful record coverage; it may even hinder coverage if per-record verbosity reduces patients-per-token throughput.
- The remaining recall limitation (2–3 missing even at 1200 best runs) is not substantially explained by this simple semantic filter; the model still fails to enumerate all 16 even when not truncated (`stop` runs at 1200: 13–14/16) and still enumerates some non-relevant despite instruction (1–4 unsupported in high-recall filtered runs).

**Do not overclaim:** We cannot claim filtering solves the bottleneck; data show filtered mean recall significantly below baseline at same budgets (non-overlapping std at 800, overlapping but lower at 1200). We cannot claim instruction has no effect – it does reduce unsupported. We cannot claim recall would improve at larger budgets – not tested.

---

## 18. Limitations

- **Single query/dataset:** Only `Give me all patients with hypertension.` (16 GT, K=20, 120 records). Other aggregates (e.g., Amlodipine 17, Paracetamol 20) not tested.
- **Two budgets only:** 800 and 1200 (per spec, 10 runs). No 200/400/600 for filtered; cannot assess curve shape below 800. Baseline showed monotonic increase to 1200; filtered only measured at top budgets where budget is less limiting.
- **Single temperature 0.3 / model llama3.2:latest / Ollama:** Sampling variance high (std 0.225 at 800). 5 reps per budget limited; bimodal distribution (4 low, 1 high at 800) suggests more reps needed for tight CI.
- **Verbosity confound:** Filtered prompt adds 51 prompt tokens and may induce longer per-patient generation, conflating filtering with token efficiency. Not separately controlled.
- **Fingerprint proxy:** Recall is *enumerated patient coverage* via age+diagnosis+etc. fingerprint (threshold ≥12). It counts a record covered if bullet contains age+hypertension; may overcount if age collides (only one colliding pair 53F) but secondary term disambiguates, or undercount if bullet truncated before diagnosis. No clinician adjudication.
- **Hallucination detection limited to age-present segments:** Non-age hallucinations not counted.
- **No human blind review, no production traffic, no latency/throughput optimization.**
- **Instruction wording fixed:** Only one phrasing tested; other wordings / placement (system vs user) may differ.

---

## 19. Conclusion

- **Retrieval frozen verified:** K=20, 16/16 relevant, context hash identical to v2, zero raw MRNs in context/prompt.
- **Only change:** filtering instruction added to prompt (recorded verbatim per run).
- **10 runs completed:** 800×5, 1200×5, Ollama metadata captured, truncation via `done_reason`.
- **Primary comparison:**

| Budget | Δ Recall (filt-baseline) | Δ Unsupported | Δ Truncation |
|--------|--------------------------|---------------|--------------|
| 800 | -0.312 (0.425 vs 0.738) | -3.0 (0.2 vs 3.2) | +0.60 (1.00 vs 0.40) |
| 1200 | -0.225 (0.600 vs 0.825) | -2.0 (1.6 vs 3.6) | 0.00 (0.60 vs 0.60) |

- **Finding:** Filtering reduces unsupported enumeration but **does not improve recall** – mean recall declines, missing increases, and at 800 truncation rate rises to 100%. Best filtered recall (0.875, 14/16) equals baseline ceiling but is rarer (1/5 vs 3/5 at 800).
- **Hypothesis verdict:** **Not supported as stated** (recall improvement not observed). **Partially supported for precision:** evidence that instruction reduces unsupported enumeration (≈2–3 fewer distractors). The remaining recall limitation is not explained by this simple filtering instruction; it may even reduce token efficiency.
- **Classification among a priori outcomes:** closest to **B** (precision improves, recall not improved) but with **negative recall delta**, indicating filtering alone is insufficient and may be counterproductive for coverage at fixed budgets.

---

## 20. Implications for the Next Experiment

- **Do NOT proceed to adaptive-K yet** (per spec) and do not change production retrieval/K/defaults based on this single result.
- **Needed next steps before adaptive-K:**

  1. **Disentangle verbosity vs filtering:** Test instruction that is **token-efficient** (e.g., "Enumerate only Hypertension records, be concise: one bullet per patient with age and Diagnosis only") to see if shorter per-patient format restores recall while keeping unsupported low. Current instruction increased tokens per patient, reducing patients-per-budget.
  2. **Test larger budgets (1500–2000):** To see if filtered recall can reach 16/16 when not budget-limited, or if saturation persists due to model reasoning limits. Current filtered 800 was always length-limited at 800 with only 5 patients; larger budget may allow 15+ patients even with verbose format.
  3. **Instruction variants:** System-prompt vs user-prompt placement, more explicit negative instruction ("Do not include CKD/Type2 Diabetes distractors"), or few-shot example of correct filtered enumeration.
  4. **Other aggregate queries:** Replicate at AGG_AMLODIPINE (17), AGG_PARACETAMOL (20) to test generalizability of precision gain vs recall loss.
  5. **Measure patients-per-token efficiency:** Report tokens per enumerated patient for filtered vs baseline to quantify verbosity cost.

- If adaptive-K proceeds, it should be **co-designed with generation filtering** (e.g., retrieve 20 but post-filter to hypertension before generation, or use K that balances recall vs distractors) and should be evaluated at budget ≥1200 to avoid confounding with truncation. The generation-filtering experiment shows budget must be ≥800–1200 for hypertension enumeration to be not purely budget-limited, and filtering alone is not a panacea.

---

## Artifacts

- `benchmarks/retrieval/generation/generation_filter_runner.py` — filtered runner (pilot + full, provider instrumentation, frozen retrieval, hash check)
- `benchmarks/retrieval/generation/generation_filter_metrics.py` — comparison helpers, `format_comparison_markdown`, delta computation
- `benchmarks/retrieval/generation/generation_filter_results.json` — 10 runs, frozen context/prompt/answers, `eval_count`/`prompt_eval_count`/`done_reason`/`truncation_status`, fingerprint metrics
- `benchmarks/retrieval/generation/generation_filter_metrics.json` — aggregate by budget (filtered)
- `reports/retrieval/generation/generation_filter_report.md` — this report
- `benchmarks/retrieval/generation/generation_results_v2.json` preserved (25 runs, hash 93eb5f...)
- `benchmarks/retrieval/generation/generation_metrics_v2.json` preserved
- `tests/test_generation_filter_metrics.py` — research-only unit tests (10 checks)

**Exact commands to reproduce:**

```bash
# Pilot only (2 runs, verifies retrieval 20/16, hash, no raw MRNs, filter present)
python3 -m benchmarks.retrieval.generation.generation_filter_runner --pilot-only

# Full 10-run experiment (includes mandatory pilot)
python3 -m benchmarks.retrieval.generation.generation_filter_runner

# Custom output path
python3 benchmarks/retrieval/generation/generation_filter_runner.py --output /tmp/custom_filter.json
```

**Exact provider/model/budget:**

- `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=llama3.2:latest`, `temperature=0.3`, `num_predict` = budget (800/1200), `stream=True` via `secure_rag/generator.py:86`.

**K=20 retrieval check:** Must show 16/16 before generation; frozen context reused; no separate post-masked index.

---

**Report generated:** 2026-09-20, `generation_filter_results.json` `generated_at` 2026-09-20T...Z, 10 generations, Ollama `llama3.2:latest`, budgets [800,1200]×5, filtered instruction per spec.
