# Generation-Budget Experiment — Controlled Report v2 (FIXED)

> **Status:** Executed 2026-09-20 — 25 runs with corrected measurand + provider instrumentation.
> **Artifacts:** `benchmarks/retrieval/generation/generation_results_v2.json`, `benchmarks/retrieval/generation/generation_metrics_v2.json`
> **v1 artifact preserved:** `generation_results_v1.json` not overwritten.

---

## 1. What was wrong with v1

**a) Invalid generation measurand for masked context:**
- Secure-RAG medical pre-embedding masking replaces raw patient IDs:
  - `Medical ID: MRN1001` → `Medical ID: [PATIENT_ID_MASKED]`
- The LLM context (frozen K=20) therefore contains **zero raw MRNs** (`MRN1001…MRN1119`). Verified in v2 context audit: `raw_mrns_in_context: []`, `has_masked_placeholder: true`.
- v1 measured generation recall by `extract_mrns()` regex for `MRN\d+` in the LLM answer. Since the LLM never receives MRNs, it cannot emit them unless it hallucinates. Result: **guaranteed 0/16** for any faithful answer. v1 reported 0.0 recall for all 25 runs despite answers actually enumerating 3–19 patients verbatim from masked chunks.
- v1 recall was therefore measuring exposure of a masked identifier, not record-level answer coverage.

**b) Ollama termination instrumentation discarded:**
- `secure_rag/generator.py:_generate_ollama` streamed `chunk["response"]` but ignored the final chunk's `done`, `done_reason`, `eval_count`, `prompt_eval_count`, `total_duration`, etc.
- v1's `_call_generation` then **inferred** truncation via heuristic: `estimated_tokens >= budget - 5`. Token counts were heuristic estimates (`words * 1.3` fallback, or local tokenizer), not provider-reported `eval_count`.
- Finish reason was set to `"unknown (streaming API does not expose finish_reason; inferred)"` unconditionally, and `truncation` was inferred boolean. Provider-supplied `done_reason` was never examined.

**c) Consequences:** v1 could not distinguish `stop` vs `length` termination, could not report provider token counts, and guaranteed invalid recall.

---

## 2. Exact definition of corrected generation measurand (v2)

**Measurand name:** *Enumerated patient/record coverage via deterministic masked-record fingerprint matching (without requiring raw MRN strings).*

**Definition:**

```
generation_recall = relevant_records_found / 16
relevant_records_found = |{ matched relevant MRNs }|
missing_records = 16 GT MRNs - matched relevant MRNs
duplicate_records = sum(count-1 for each GT MRN with count>1 in answer)
unsupported_records = matched non-relevant retrieved MRNs + hallucinated patient-like segments
```

- Ground truth relevant set remains the frozen 16 hypertension MRNs:
  `MRN1001, MRN1005, MRN1021, MRN1025, MRN1026, MRN1052, MRN1059, MRN1066, MRN1070, MRN1074, MRN1085, MRN1104, MRN1107, MRN1111, MRN1118, MRN1119`.
- Retrieved candidate pool remains the frozen K=20 (order-fixed, reused every run):
  `MRN1107, MRN1066, MRN1104, MRN1025, MRN1059, MRN1021, MRN1111, MRN1119, MRN1026, MRN1085, MRN1052, MRN1070, MRN1001, MRN1074, MRN1005, MRN1118, MRN1054, MRN1094, MRN1015, MRN1051`
  (16 relevant + 4 non-relevant: `MRN1054` CKD, `MRN1094` CKD, `MRN1015` T2D, `MRN1051` T2D).
- The LLM answer is split into **enumerated patient segments** (numbered list `1. … 2. …`). Each segment must contain an age phrase (`\d+-year-old (male|female)` or `\d+-year-old`) to be considered a patient mention. Fallback: if no numbered list, each age mention window is a segment.
- Each segment is scored deterministically against **all 20 frozen masked-record fingerprints** (see §3). Best score >=12 is assigned to that record. This yields `matched_ordered_all` and `counter_by_id`.
- `relevant_records_found` counts distinct GT MRNs matched at least once.
- `duplicate_records` counts extra mentions beyond first per GT MRN.
- `unsupported_records` = matched non-relevant retrieved IDs plus hallucinated segments (age present but no fingerprint score >=12).
- `enumerated_patient_count` = number of patient-like segments (age present) in answer.
- `hallucinated_count` = patient-like segments with no fingerprint match.

**Implementation:** `benchmarks/retrieval/generation/generation_metrics.py:compute_masked_generation_metrics`, `build_masked_record_fingerprints`, `_split_answer_into_patient_segments`, `_score_segment_against_fingerprint`.

**Recall semantics:** Fraction of the 16 relevant hypertension masked records that are *represented* in the generated answer as an enumerated patient entry containing that record's stable visible attributes. No raw MRN string is required.

---

## 3. How the measurand works despite masked MRNs

- **Production masking unchanged:** `Medical ID: [PATIENT_ID_MASKED]` remains; raw MRNs never enter LLM context. Verified by context audit (`raw_mrn_count: 0` for every run).
- **Deterministic external mapping:** Outside the LLM, the experiment builds `chunk_index → MRN` mapping before embedding:
  - `mask_text()` with `medical` policy + detector stack → masked chunk per record → `chunk_record_map[i] = MRN` before chunking/embedding.
  - Retrieval at K=20 yields `frozen_retrieved_ids` in order; `context_chunks[i]` is the masked chunk for `frozen_retrieved_ids[i]`.
  - This mapping is passed to `compute_masked_generation_metrics` as parallel arrays `retrieved_chunks` and `retrieved_ids`. The LLM never sees MRNs.
- **Fingerprint construction from visible attributes:**
  Each masked chunk yields a fingerprint derived only from content the LLM sees:
  - `age`: `\d+-year-old (male|female)` (e.g., `53-year-old female`)
  - `age_num`: `\d+-year-old` (e.g., `53-year-old`) — allows gender hallucination tolerance
  - `diagnosis`: `Diagnosis: …` line (e.g., `Hypertension; Type 2 Diabetes; Iron Deficiency Anaemia`)
  - `treatment`: `Treatment: …` line (e.g., `Telmisartan 40mg, Low-salt diet`)
  - `admission`: `was admitted … with …` reason phrase
  - `norm_*` lowercased/normalized variants
  See `build_masked_record_fingerprints`.
- **Segment scoring (auditable, deterministic):**
  ```
  +10 age_num present (mandatory)
   +2 exact age+gender present
   +5 first diagnosis term present (e.g., "hypertension")
   +3 full diagnosis snippet 40-char present
   +5 each secondary diagnosis term present (e.g., "iron deficiency anaemia" distinguishes MRN1104 53F-anaemia from MRN1107 53F-hypertension-only)
   +2 first treatment drug present (e.g., "telmisartan 40mg")
   +2 admission reason phrase present
  ```
  Threshold `>=12` required to assign. This distinguishes:
  - `MRN1107` (53F, Hypertension, Telmisartan) vs `MRN1104` (53F, Hypertension+anaemia, [NAME_MASKED] 50mg) via anaemia term.
  - `MRN1025` (56F) vs `MRN1015` (56M T2D) via admission reason `elevated blood pressure` vs `excessive thirst`.
- **No LLM-side identifier invented:** Matching uses only masked-record content that survives redaction.

**Example:** Answer bullet `1. [NAME_MASKED], a 53-year-old female patient, was admitted … with persistent hypertension.` scores 10 (age_num) +2 (gender) +5 (hypertension) +2 (admission) =19 → matched to `MRN1107`. Bullet `3. … with persistent hypertension; Type 2 Diabetes; Iron Deficiency Anaemia.` scores +5+5 for anaemia → best match `MRN1104` despite sharing age 53F with MRN1107.

**Limitations documented:** This is an *enumerated patient/record coverage* proxy, not perfect semantic entailment. It counts a record as covered if its enumerated bullet contains age+diagnosis overlap. It may:
- Mark a bullet with hallucinated gender (e.g., 56F rendered as 56M male) as covering the 56F record if age_num+diagnosis still matches (lenient to gender error).
- Overcount if diagnosis alone is generic (`Hypertension`) and age collides (only one colliding pair: two 53F records, disambiguated by anaemia term; all other relevant records have distinct age_num).
- Undercount if bullet is truncated before diagnosis text appears (e.g., budget_200 third bullet truncated at `Diagnosis:` with no value → not counted).
- Not evaluate correctness of treatment details beyond first drug presence.

---

## 4. Exact Ollama metadata now captured

**File:** `secure_rag/generator.py:_generate_ollama` and `get_last_ollama_metadata()`.

Previously discarded; now captured separately while preserving production streaming behavior (text still yielded exactly as before via `yield chunk["response"]`).

Captured fields from final streaming chunk(s) (where present, missing handled safely):

| Field | Source | Stored in |
|-------|--------|-----------|
| `done` | `chunk["done"]` | `ollama_metadata["done"]` |
| `done_reason` | `chunk["done_reason"]` | `ollama_metadata["done_reason"]`, `finish_reason` / `done_reason` per-run |
| `eval_count` | `chunk["eval_count"]` | `ollama_metadata["eval_count"]`, `output_tokens` when available |
| `prompt_eval_count` | `chunk["prompt_eval_count"]` | `ollama_metadata["prompt_eval_count"]`, `prompt_tokens` when available |
| `total_duration` | `chunk["total_duration"]` | `ollama_metadata["total_duration"]` |
| `load_duration` | `chunk["load_duration"]` | `ollama_metadata["load_duration"]` |
| `prompt_eval_duration` | `chunk["prompt_eval_duration"]` | `ollama_metadata["prompt_eval_duration"]` |
| `eval_duration` | `chunk["eval_duration"]` | `ollama_metadata["eval_duration"]` |
| `model` | `chunk["model"]` | `ollama_metadata["model"]` |
| `created_at` | `chunk["created_at"]` | `ollama_metadata["created_at"]` |

Missing fields handled safely: not assumed present. Final metadata dict is stored in global `_LAST_OLLAMA_METADATA` after stream exhaustion and exposed via `get_last_ollama_metadata()` to the research runner without changing production `generate_answer` signature.

**Example captured (budget 800 rep1, truncated stop):**
```json
{
  "done": true,
  "done_reason": "stop",
  "eval_count": 705,
  "prompt_eval_count": 3415,
  "total_duration": 5209500000,
  "load_duration": 225000000,
  "prompt_eval_duration": 287000000,
  "eval_duration": 210000000
}
```

---

## 5. Exact truncation methodology (v2)

**Old (deleted):** `estimated_tokens >= budget - 5` heuristic.

**New (provider-authoritative):**

```python
done_reason = ollama_metadata.get("done_reason")
if done_reason == "length":
    finish_reason = "length"
    truncation = True
    truncation_status = "truncated"
elif done_reason == "stop":
    finish_reason = "stop"
    truncation = False
    truncation_status = "not_truncated"
else:  # missing or other
    finish_reason = "unknown"  # or exact done_reason string
    truncation = None
    truncation_status = "unknown"
```

- `truncation` is `bool | None` (None → unknown, not False).
- `truncation_status` distinguishes `truncated` / `not_truncated` / `unknown`.
- Unknown is **not silently converted to false**. Aggregate reporting counts `unknown_termination_count` separately.

---

## 6. Pilot results (mandatory 2-run pilot, budgets 200 & 1200, rep 1)

Executed via `benchmarks/retrieval/generation/generation_budget_runner.py:run_pilot` before any full runs.

| Check | Expected | Pilot 200 | Pilot 1200 | Verdict |
|-------|----------|-----------|------------|---------|
| 1. Retrieval 16/16 | K=20, 16 relevant | 20/16 | 20/16 | PASS |
| 2. Frozen context identical | Reused | yes | yes | PASS |
| 3. Raw MRNs NOT in LLM context | 0 raw | 0 raw, `[PATIENT_ID_MASKED]` present | 0 raw | PASS |
| 4. Model output contains patient info evaluable | answer len >20 | 699 chars | 4314 chars | PASS |
| 5. New metric non-trivial | recall >0 and <1 | 0.188 (3/16) enum 3 | 0.812 (13/16) enum 16 | PASS |
| 6. Ollama final metadata captured | non-empty dict | `{done:True, done_reason:length, eval_count:200,…}` | `{done:True, done_reason:length, eval_count:1200,…}` | PASS |
| 7. eval_count captured | present | 200 | 1200 | PASS |
| 8. done_reason captured | `length`/`stop` | `length` | `length` | PASS |
| 9. Truncation based on provider | `done_reason=="length"` → truncated | `truncated`/`length` | `truncated`/`length` | PASS (not heuristic) |
| 10. 200 vs 1200 distinguishable | tokens or finish or recall differ | tokens 200≠1200, recall 3≠13 | — | PASS |

Pilot 200: `finish_reason=length`, `truncation=truncated`, `output_tokens=200` (`is_estimated=false`), latency 48764ms, answer truncated mid-`Diagnosis:` at third bullet.

Pilot 1200: `finish_reason=length`, `truncation=truncated`, `output_tokens=1200`, latency ~111s, answer enumerated 16 patients (13 relevant) but still hit `length` (budget-limited) rather than `stop`.

**Pilot STOP condition:** All 10 checks passed → proceeded to 23 remaining generations.

---

## 7. Full 25-run results (individual)

Provider `ollama`, model `llama3.2:latest`, temperature `0.3`, query `Give me all patients with hypertension.`, K=20, 16/16 relevant frozen.

| run_id | budget | recall | found | missing | dup | unsupported | enum | halluc | trunc status | finish | out_tokens | est | prompt_tok | latency_ms |
|--------|--------|--------|-------|---------|-----|-------------|------|--------|--------------|--------|------------|-----|------------|------------|
| budget_200_rep_1 | 200 | 0.188 | 3 | 13 | 0 | 0 | 3 | 0 | truncated | length | 200 | false | 3415 | 48764 |
| budget_200_rep_2 | 200 | 0.188 | 3 | 13 | 1 | 2 | 6 | 0 | truncated | length | 200 | false | 3415 | 15306 |
| budget_200_rep_3 | 200 | 0.188 | 3 | 13 | 1 | 1 | 5 | 0 | truncated | length | 200 | false | 3415 | 14569 |
| budget_200_rep_4 | 200 | 0.188 | 3 | 13 | 0 | 0 | 3 | 0 | truncated | length | 200 | false | 3415 | 15771 |
| budget_200_rep_5 | 200 | 0.125 | 2 | 14 | 0 | 1 | 3 | 0 | truncated | length | 200 | false | 3415 | 17812 |
| budget_400_rep_1 | 400 | 0.562 | 9 | 7 | 1 | 1 | 11 | 0 | truncated | length | 400 | false | 3415 | 29442 |
| budget_400_rep_2 | 400 | 0.312 | 5 | 11 | 0 | 0 | 5 | 0 | truncated | length | 400 | false | 3415 | 32573 |
| budget_400_rep_3 | 400 | 0.312 | 5 | 11 | 0 | 0 | 5 | 0 | truncated | length | 400 | false | 3415 | 30401 |
| budget_400_rep_4 | 400 | 0.562 | 9 | 7 | 1 | 1 | 11 | 0 | truncated | length | 400 | false | 3415 | 30724 |
| budget_400_rep_5 | 400 | 0.312 | 5 | 11 | 0 | 0 | 5 | 0 | truncated | length | 400 | false | 3415 | 29231 |
| budget_600_rep_1 | 600 | 0.500 | 8 | 8 | 0 | 0 | 8 | 0 | truncated | length | 600 | false | 3415 | 43628 |
| budget_600_rep_2 | 600 | 0.500 | 8 | 8 | 0 | 0 | 8 | 0 | truncated | length | 600 | false | 3415 | 48722 |
| budget_600_rep_3 | 600 | 0.500 | 8 | 8 | 0 | 0 | 8 | 0 | truncated | length | 600 | false | 3415 | 48418 |
| budget_600_rep_4 | 600 | 0.500 | 8 | 8 | 0 | 0 | 8 | 0 | truncated | length | 600 | false | 3415 | 52042 |
| budget_600_rep_5 | 600 | 0.812 | 13 | 3 | 1 | 3 | 17 | 0 | truncated | length | 600 | false | 3415 | 49802 |
| budget_800_rep_1 | 800 | 0.812 | 13 | 3 | 1 | 5 | 19 | 0 | not_truncated | stop | 705 | false | 3415 | 52095 |
| budget_800_rep_2 | 800 | 0.625 | 10 | 6 | 0 | 1 | 11 | 0 | truncated | length | 800 | false | 3415 | 81975 |
| budget_800_rep_3 | 800 | 0.625 | 10 | 6 | 0 | 1 | 11 | 0 | truncated | length | 800 | false | 3415 | 88689 |
| budget_800_rep_4 | 800 | 0.812 | 13 | 3 | 2 | 4 | 19 | 0 | not_truncated | stop | 707 | false | 3415 | 86459 |
| budget_800_rep_5 | 800 | 0.812 | 13 | 3 | 1 | 5 | 19 | 0 | not_truncated | stop | 707 | false | 3415 | 77996 |
| budget_1200_rep_1 | 1200 | 0.812 | 13 | 3 | 0 | 3 | 16 | 0 | truncated | length | 1200 | false | 3415 | 111010 |
| budget_1200_rep_2 | 1200 | 0.812 | 13 | 3 | 0 | 3 | 16 | 0 | truncated | length | 1200 | false | 3415 | 105792 |
| budget_1200_rep_3 | 1200 | 0.812 | 13 | 3 | 0 | 3 | 16 | 0 | truncated | length | 1200 | false | 3415 | 103672 |
| budget_1200_rep_4 | 1200 | 0.875 | 14 | 2 | 0 | 4 | 18 | 0 | not_truncated | stop | 1113 | false | 3415 | 100218 |
| budget_1200_rep_5 | 1200 | 0.812 | 13 | 3 | 1 | 5 | 19 | 0 | not_truncated | stop | 707 | false | 3415 | 73563 |

Raw `answer_text` for each run stored verbatim in `generation_results_v2.json` (context, prompt, `ollama_metadata` included).

---

## 8. Aggregate results by generation budget

Source `generation_metrics_v2.json` via `benchmarks/retrieval/generation/generation_metrics.py:aggregate_by_budget` (now handles tri-state truncation).

| budget | runs | mean_recall | std_recall | min–max | mean_missing | std_missing | mean_dup | mean_unsupported | trunc_rate | trunc_count | unknown_count | mean_out_tokens | mean_prompt_tokens | mean_latency_ms |
|--------|------|-------------|------------|---------|--------------|-------------|----------|------------------|------------|-------------|---------------|-----------------|------------------|-----------------|
| 200 | 5 | 0.175 | 0.025 | 0.125–0.188 | 13.2 | 0.40 | 0.4 | 0.8 | 1.00 | 5 | 0 | 200.0 | 3415.0 | 22444.4 |
| 400 | 5 | 0.412 | 0.122 | 0.312–0.562 | 9.4 | 1.96 | 0.4 | 0.4 | 1.00 | 5 | 0 | 400.0 | 3415.0 | 30474.2 |
| 600 | 5 | 0.562 | 0.125 | 0.500–0.812 | 7.0 | 2.00 | 0.2 | 0.6 | 1.00 | 5 | 0 | 600.0 | 3415.0 | 48522.4 |
| 800 | 5 | 0.737 | 0.092 | 0.625–0.812 | 4.2 | 1.47 | 0.8 | 3.2 | 0.40 | 2 | 0 | 743.8 | 3415.0 | 77442.8 |
| 1200 | 5 | 0.825 | 0.025 | 0.812–0.875 | 2.8 | 0.40 | 0.2 | 3.6 | 0.60 | 3 | 0 | 1084.0 | 3415.0 | 98851.0 |

- `trunc_rate` = `truncated / runs` (unknown excluded). `unknown_termination_*` = 0 for all budgets (provider always returned `done_reason`).
- `mean_out_tokens` = mean `eval_count` (provider-reported, `is_estimated=false` for all 25 runs).
- `mean_prompt_tokens` = mean `prompt_eval_count` 3415 (provider-reported, `is_estimated=false`).

---

## 9. Generation recall vs generation budget

- **200:** 0.175 ±0.025 (2–3/16). All 5 runs `length` truncated at budget (200 tokens). Third bullet truncated mid-`Diagnosis:`; only first ~2.5 patients enumerated.
- **400:** 0.412 ±0.122 (5–9/16). Bimodal: two runs 0.562 (9/16) enumerating ~11 patients, three runs 0.312 (5/16) enumerating ~5. All still `length` truncated.
- **600:** 0.562 ±0.125 (8–13/16). Four runs 0.500 (8/16) via 8 enumerated patients; one outlier 0.812 (13/16) with 17 enumerated but still `length` truncated.
- **800:** 0.737 ±0.092 (10–13/16). Three runs reached `stop` (not truncated) at ~705–707 tokens, recall 0.812. Two runs still `length` at 800 tokens, recall 0.625.
- **1200:** 0.825 ±0.025 (13–14/16). Three runs `length` at 1200 tokens recall 0.812; two runs `stop` at 707 and 1113 tokens recall 0.812–0.875. No run reached 16/16.

Recall increases monotonically with budget (175→412→562→737→825) but **saturates well below 1.0**. Even at 1200, mean recall 0.825 (13.2/16 average). Maximum observed 14/16 (0.875) in `budget_1200_rep_4`; minimum at 200 is 0.125 (2/16). No evidence that larger budget alone achieves 16/16.

---

## 10. Missing / duplicate / unsupported records vs budget

| budget | mean_missing | mean_duplicate | mean_unsupported | enumerated_patient_count (mean) |
|--------|--------------|----------------|------------------|---------------------------------|
| 200 | 13.2 | 0.4 | 0.8 | 4.0 |
| 400 | 9.4 | 0.4 | 0.4 | 7.4 |
| 600 | 7.0 | 0.2 | 0.6 | 9.8 |
| 800 | 4.2 | 0.8 | 3.2 | 15.8 |
| 1200 | 2.8 | 0.2 | 3.6 | 17.0 |

- **Missing** decreases with budget (13.2 → 2.8) as expected, but plateaus at ~2.8 missing even at 1200.
- **Duplicate** remains low (<1) at all budgets; occasional duplicate when model repeats a patient (e.g., 200 rep2 duplicated one 53F). No systematic increase.
- **Unsupported** (non-relevant enumerated) increases with budget: 0.8 at 200 → 3.6 at 1200. At larger budgets the model enumerates **all 20 retrieved records**, including the 4 non-hypertension distractors (`MRN1054` CKD, `MRN1094` CKD, `MRN1015` T2D, `MRN1051` T2D). Example: 1200 rep1 enumerated 16 patients including 3 unsupported (the 3 non-relevant plus one hallucinated). This suggests the LLM does not perfectly filter to hypertension-only when enumerating, and larger budgets allow more unsupported enumeration to surface.
- **Enumerated count** tracks budget: 4 → 7.4 → 9.8 → 15.8 → 17.0, approaching K=20 at 800/1200.

---

## 11. Actual truncation / termination behavior vs budget

- **Token source:** All 25 runs have `output_tokens = eval_count` with `output_tokens_is_estimated=false` and `prompt_tokens = prompt_eval_count = 3415` with `prompt_tokens_is_estimated=false`. No heuristic estimation used.
- **Finish reasons observed:** Only `length` and `stop`. No `unknown`.
  - `length` = generation hit `num_predict` budget (`done_reason==length`) → `truncated`.
  - `stop` = model stopped naturally before budget (`done_reason==stop`) → `not_truncated`.
- **Per-budget truncation:**
  - 200: 5/5 truncated (100%) – all hit `length` at 200 tokens.
  - 400: 5/5 truncated (100%) at 400.
  - 600: 5/5 truncated (100%) at 600.
  - 800: 2/5 truncated (40%) at 800, 3/5 stopped early (705–707 tokens).
  - 1200: 3/5 truncated (60%) at 1200, 2/5 stopped early (707, 1113 tokens).
- **Unknown terminations:** 0/25. All runs returned usable `done_reason`.

**Observation:** Truncation dominates at 200–600 (all length-limited). At 800, some runs finish early (`stop`) at ~706 tokens with recall 0.812, indicating the model sometimes decides it has enumerated enough even though 13/16 is not 16/16. At 1200, 3 runs still hit length limit (1200) while enumerating 16 patients (recall 0.812), showing enumeration of ~16 patients can still require 1200 tokens when including full masked record text per patient.

---

## 12. Output tokens vs generation budget

| budget | mean_output_tokens (eval_count) | budget utilization | was truncated? |
|--------|----------------------------------|--------------------|----------------|
| 200 | 200.0 | 100% | yes |
| 400 | 400.0 | 100% | yes |
| 600 | 600.0 | 100% | yes |
| 800 | 743.8 | 92.9% | 40% truncated |
| 1200 | 1084.0 | 90.3% | 60% truncated |

- At 200–600, `eval_count == budget` exactly → `length` termination; generation was budget-limited.
- At 800, `eval_count` 705–800 (mean 743.8) < budget for 3 runs → `stop`; those runs used 88% of budget on average.
- At 1200, `eval_count` 707–1200 (mean 1084) < budget for 2 runs → `stop`; 3 runs still exhausted budget.

This confirms provider-reported token accounting (not estimated) and shows larger budgets are not always fully utilized when model decides to stop.

---

## 13. Latency vs generation budget

| budget | mean_latency_ms | latency per token (approx) |
|--------|-----------------|----------------------------|
| 200 | 22444 | ~112 ms/tok |
| 400 | 30474 | ~76 ms/tok |
| 600 | 48522 | ~81 ms/tok |
| 800 | 77443 | ~104 ms/tok |
| 1200 | 98851 | ~91 ms/tok |

Latency increases monotonically with budget/output tokens (22s → 98s). Not strictly linear due to Ollama scheduling variance, but larger budgets consistently take longer (2–4×). `total_duration` from Ollama metadata corroborates wall-clock.

---

## 14. Interpretation

- **Generation budget is a real bottleneck, but not the sole bottleneck.** Recall improves strongly from 200→400→600→800 (+0.237, +0.150, +0.175), then plateaus at 800→1200 (+0.088). At 800, recall already 0.737 (11.8/16) and at 1200 0.825 (13.2/16). The improvement curve is monotonic but diminishing.
- **Even with 1200 tokens, no run achieved 16/16.** Maximum 14/16. The model’s enumeration behavior at 800/1200 often includes 4 non-relevant records, suggesting instruction following / filtering is imperfect, not purely budget-limited.
- **Truncation is not strictly budget-deterministic at 800/1200.** At 800, 60% of runs stopped early (`stop`) despite recall 0.812, indicating the model believed it had answered fully (enumerated ~16–19 patients) without needing the full budget. At 1200, 40% stopped early. This suggests the LLM’s internal stopping criterion saturates before exhaustive enumeration.
- **Unsupported enumeration grows with budget.** The ability to enumerate more patients with larger budgets also surfaces distractors, increasing unsupported from 0.8→3.6. A larger budget alone does not improve precision.
- **Duplicate low.** Model rarely repeats same patient verbatim; occasional duplicate (0.4 mean) suggests no systematic duplication issue.

**Neutral statement:** Increasing `num_predict` from 200 to 1200 increases mean recall from 0.175 to 0.825 (+0.65) and reduces missing from 13.2→2.8, but recall saturates around 13–14/16. The budget explains much of the incompleteness at 200–600 (all truncated), but at 800–1200 where some runs are not truncated, remaining incompleteness (2–3 missing) is not solely budget.

---

## 15. Limitations

- **Single query/dataset:** Only `Give me all patients with hypertension.` (16 GT, K=20) with masked medical records. Generalization to other aggregate queries (e.g., `Amlodipine 5mg` 17 GT, `Paracetamol 650mg` 20 GT) not tested.
- **Single temperature (0.3) and provider/model (Ollama llama3.2:latest):** Sampling variance limited to 5 reps per budget; other temperatures/models may differ.
- **Enumerated patient proxy:** Measurand is *enumerated patient/record coverage* via deterministic age+diagnosis fingerprint, not full semantic equivalence or clinical correctness of treatment details. It requires age phrase presence and diagnosis overlap score ≥12. Truncated bullets missing diagnosis are not counted (may undercount attempted enumeration). Gender hallucination (e.g., 56F rendered as 56M) is tolerated if age_num+diagnosis matches (may overcount). See §3 for scoring details.
- **Hallucination vs unsupported:** Hallucinated patient-like segments (age present, no fingerprint match) counted in `unsupported_count` via `hallucinated_count`, but model could hallucinate entirely non-existent patient without age phrase → not counted.
- **Embedding/retrieval frozen at seed 42:** Retrieval depth 16/16 at K=20 is deterministic for this seed; other seeds may differ.
- **No human adjudication:** Coverage judgment is deterministic regex/fingerprint, not clinician review.
- **Prompt fixed:** Uses production Secure-RAG prompt (`You are a RAG assistant...`). Other prompts may yield different enumeration behavior.

---

## 16. Whether the generation-budget hypothesis is supported

**Hypothesis:** *If the generation budget is the bottleneck, larger budgets should increase record-level generation recall toward 16/16, reduce missing, and correlate with reduced truncation.*

**Verdict: Partially supported, but not fully.**

- **Supported:** Recall increases monotonically with budget (0.175→0.825) and missing decreases (13.2→2.8). Truncation rate is 1.0 at 200–600 and drops to 0.4–0.6 at 800–1200, correlating with recall gain. At 200–600, all runs are `length` truncated and recall is low, indicating budget limits enumeration (~3–8 patients enumerated at 200–600 vs ~16–19 at 800–1200).
- **Not fully:** Recall saturates at ~0.825 (13/16) even at 1200, and 3/5 runs at 1200 are still `length` truncated at 1200 tokens while enumerating only 16 patients (including unsupported). Two runs at 1200 stopped early (`stop`) at 707 and 1113 tokens with recall 0.812–0.875, indicating not all incompleteness is budget. The remaining 2–3 missing records at 1200 are not explained by truncation alone (since some `stop` runs still miss 3). The model also increasingly enumerates non-relevant records, so budget increase improves recall but degrades precision.

**Precise claim warranted:** Generation budget is a *significant* bottleneck for budgets ≤600 (all truncated, recall ≤0.56). At 800–1200, budget is *partially* the bottleneck, but recall saturates below exhaustive 16/16, suggesting additional limitations (enumeration completeness, filtering to relevant only, or model instruction-following saturates).

Do **not** claim that `1200` solves the problem or that larger budget alone achieves 16/16; data do not support.

---

## 17. Whether adaptive-K should proceed next

**Recommendation: Yes, but after addressing generation filtering, not just budget.**

- Retrieval at K=20 already achieves 16/16, but generation never reaches 16/16 even when not truncated. Adaptive-K (e.g., dynamic K based on query type) may reduce unsupported enumeration (currently 3.6 at 1200) by retrieving fewer distractors, but will not alone solve recall saturation.
- More promising next steps before pure adaptive-K:
  1. **Improve generation filtering instruction** to enumerate only relevant (hypertension) among the 20, not all retrieved — currently unsupported grows with budget.
  2. **Test larger budgets >1200** (e.g., 1500–2000) to see if recall can reach 16/16 when not truncated, or if saturation persists due to model reasoning limits.
  3. **Evaluate other aggregate queries** (e.g., `ACG_AMLODIPINE_5MG` 17 GT, `AGG_PARACETAMOL` 20 GT) to see if saturation pattern replicates.
- If adaptive-K proceeds, design should: retrieve 20 but prompt the model to enumerate only records where `Diagnosis` contains `Hypertension` (or use post-retrieval filtering before generation), and measure recall vs K curve (the retrieval depth experiment). The generation-budget experiment shows budget must be at least 800–1200 for hypertension enumeration to be not budget-limited; any adaptive-K evaluation should fix budget ≥800 to avoid confounding.

**Do not yet prioritize adaptive-K as the sole next experiment without also investigating generation instruction/filtering and testing recall ceiling.**

---

## Artifacts

- `benchmarks/retrieval/generation/generation_results_v2.json` — 25 runs, frozen context/prompt/answers, provider metadata (`eval_count`, `prompt_eval_count`, `done_reason`, `truncation_status`, `output_tokens_is_estimated=false`), masked-record metrics.
- `benchmarks/retrieval/generation/generation_metrics_v2.json` — aggregate by budget (mean/std/min/max recall, missing, duplicate, unsupported, truncation/unknown counts, mean tokens/latency).
- `benchmarks/retrieval/generation/generation_budget_runner.py` — v2 runner (pilot + full, provider instrumentation, masked measurand).
- `benchmarks/retrieval/generation/generation_metrics.py` — v2 metrics (masked fingerprints, `compute_masked_generation_metrics`, tri-state aggregation).
- `secure_rag/generator.py` — Ollama streaming instrumentation (`get_last_ollama_metadata`, `done`/`done_reason`/`eval_count` capture).
- This report: `reports/retrieval/generation/generation_budget_report_v2.md`

**Exact command to reproduce:**
```bash
python3 -m benchmarks.retrieval.generation.generation_budget_runner
# or pilot-only
python3 -m benchmarks.retrieval.generation.generation_budget_runner --pilot-only
# custom output
python3 benchmarks/retrieval/generation/generation_budget_runner.py --output /tmp/custom.json
```

**Exact provider/model/budget parameter:**
- Provider: `LLM_PROVIDER=ollama` (default)
- Model: `OLLAMA_MODEL=llama3.2:latest` (via `os.getenv("OLLAMA_MODEL", "llama3.2")`)
- Generation parameter: `num_predict` (and top-level `num_predict`) = generation budget, `temperature=0.3`, `stream=True` via `secure_rag/generator.py:86`.

**K=20 retrieval check:** Must show 16/16 before generation (experiment fails otherwise). Frozen context reused for every run; no separate post-masked index.

---

**Report generated:** 2026-09-20, `generation_results_v2.json` `generated_at` 2026-09-20T05:52:00Z, 25 generations, Ollama `llama3.2:latest`, budgets [200,400,600,800,1200]×5.

