# Aggregate Retrieval Failure — Runtime Audit

**Date:** 2026-09-17  
**Query audited:** `list all the patients admitted with hypertension` (audit query) + canonical `Which patients have Hypertension?` (`AGG_HYPERTENSION`)  
**Dataset:** `data/sample_patient_data.txt` — 120 records, 16 Hypertension (verified)  
**Code at HEAD:** `master` clean (`git status` clean before audit; no prod files modified)

---

## A. Executive Finding

**The interactive runtime does NOT use k=30/50. It uses k=2.**

- `secure_rag/retriever.py:4` defaults `k=2`; `secure_rag/rag_pipeline.py:96` calls `retrieve(query, vector_store, chunks)` with **no explicit `k`** → inherits `k=2`.
- `secure_rag/vector_store.py:17` also defaults `k=2` and caps via `min(k, ntotal)`.
- The belief that interactive runs tested `k=30/50` is a **configuration/runtime mismatch**. The benchmark depth experiment (`benchmarks/retrieval/runner.py:24` `MAX_K=50, K_VALUES=[1,3,5,10,20,30,50]`, `_retrieve_top_k(..., k=50)` at `runner.py:194`) explicitly passes `k`, so its reported `k=20/30/50 → 16/16 recall` is correct **for the benchmark code path**, but was never exercised by the interactive `build_rag → rag_answer` path.

Consequence: every interactive `secure-rag <file>` session since `HEAD` has returned **2 chunks** to the LLM, giving `2/16 recall (0.125)` for the hypertension aggregate. The reported contradictory outcomes (15 patients vs “no information”) cannot be reproduced from the current committed runtime at any fixed K alone — they imply (a) a dirty working tree where `retriever.py` was locally changed to `k=50` (observed as an uncommitted diff during this audit, restored), and/or (b) **generation-stage** truncation / nondeterminism on top of retrieval.

Retrieval itself is **not broken** when given sufficient K: at `k≥20` both queries achieve `16/16 recall` on the current masked index. The failure is a **depth-insufficiency + generation-budget** problem, not masking, not embedding, not FAISS.

---

## B. Exact Runtime Retrieval Path

```
CLI: secure_rag.cli:main → chat(file_path)               # cli.py:252-253, typer entry
  → build_rag(file_path)                                  # rag_pipeline.py:66
      load_data(file_path)                                # rag_pipeline.py:48 (clean_input_text :17, split)
      split_into_records(text)                            # pdf_loader.py:18  (text.split("\n\n"))
      for record in records:
          mask_text(record, policy=medical, detectors=medical)  # rag_pipeline.py:75, masker.py:5
          chunks.extend(chunk_record(record))             # rag_pipeline.py:77, pdf_loader.py:27 (chunk_size=500 → 1 chunk/record here)
      embed_chunks(chunks)                                # embedding.py:23 (all-MiniLM-L6-v2 → float32)
      VectorStore(embeddings)                             # vector_store.py:6 (faiss.IndexFlatL2 + add)
  → retrieve(query, vector_store, chunks)                 # rag_pipeline.py:96  *** NO k ARG ***
      → embed_chunks([query])                             # retriever.py:6  (query never masked)
      → vector_store.search(query_vector, k)              # retriever.py:9 → vector_store.py:22 (limit=min(k,ntotal), IndexFlatL2.search)
      → return [chunks[i].strip() ...]                    # retriever.py:11
  → context = "\n\n".join(chunk for chunk in context_chunks if chunk)  # rag_pipeline.py:97
  → generate_answer(context, f"{query}\n\nAnswer:")       # rag_pipeline.py:99 → generator.py:54
      → if LLM_PROVIDER==ollama: _generate_ollama(...)    # generator.py:30 (http://localhost:11434/api/generate, model=llama3.2, stream=True)
        else: OpenAI(HF router) chat.completions.create(  # generator.py:118  model=Qwen/Qwen2.5-7B-Instruct, max_tokens=200, temperature=0.3, stream=True
              messages=[system prompt 62-110, user f"Context:\n{context}\n\nQuestion:\n{query}"])
  → _truncate_at_stop_marker(response)                    # rag_pipeline.py:87 (cuts at \nContext:/\nQuestion:/[/INST])
  ← CLI streams via Live(Markdown) + stream_string         # cli.py:222-226
```

No query masking anywhere. All retrieved chunks are passed unfiltered; no dedup, no token budget, no truncation in `rag_answer` itself.

---

## C. Actual K Verification

| Location | File:Line | Signature | Default | How invoked |
|---|---|---|---|---|
| **Runtime retriever** | `retriever.py:4` | `def retrieve(query, vector_store, chunks, k=2)` | **2** | `rag_pipeline.py:96` calls without `k` → **inherits 2** |
| VectorStore fallback | `vector_store.py:17` | `def search(self, query_vector, k=2)` | 2 | `limit = min(k, self.index.ntotal)` :22 |
| Benchmark runner | `retrieval/runner.py:56` | `def _retrieve_top_k(query, vs, k)` | required | called as `k=50` (`runner.py:194`) |
| Benchmark privacy | `_common.py:29` | `RETRIEVAL_K = 5` | 5 | `privacy_eval.py:209` separate leakage test |
| Depth experiment | `depth_experiment.py:15` | `K_CANDIDATES=[2,5,10,20,30,50]` | — | slices `retrieved[:k]` from `k=50` result |

**Evidence of default:**

```python
# /tmp/audit_retrieval.py
inspect.signature(retrieve).parameters['k'].default  # → 2
default_chunks = retrieve(query, vector_store, chunks)  # no k → len==2
# → "Default retrieve returned 2 chunks" (proven)
```

```python
# secure_rag/rag_pipeline.py:95-96
def rag_answer(query: str, vector_store, chunks):
    context_chunks = retrieve(query, vector_store, chunks)  # ← no k argument
```

**Uncommitted-dirty observation (restored):** at audit start, `git diff` showed `retriever.py` locally changed `k=2 → k=50`. This explains how an interactive run could have produced `k=50` behavior while `HEAD` remains `k=2`. After `git restore`, runtime reverted to `k=2` — the committed baseline this audit measures.

The contradiction `retriever.py k=50 vs vector_store.py k=2` described in the task is a misread: **both default to `k=2` at HEAD**; the `k=50` was a local dirty edit, not committed code.

---

## D. Retrieval Results for k=2/10/20/30/50

Method: `/tmp/audit_retrieval.py` — builds fresh masked index via `build_rag("data/sample_patient_data.txt")` (120 chunks, 1:1), embeds both queries with `all-MiniLM-L6-v2`, searches via `VectorStore.search` (L2), maps chunks→MRN via reconstructed `chunk_record_map`, deduplicates record IDs.

**Hypertension ground truth:** 16 records — `MRN1001, MRN1005, MRN1021, MRN1025, MRN1026, MRN1052, MRN1059, MRN1066, MRN1070, MRN1074, MRN1085, MRN1104, MRN1107, MRN1111, MRN1118, MRN1119` (matches `AGG_HYPERTENSION` + dataset scan).

### Audit query: `list all the patients admitted with hypertension`

| K | Retrieved (unique records) | Hypertension retrieved | Recall | Precision | Note |
|---|---|---|---|---|---|
| 2 | 2 | 2/16 | 0.125 | 1.000 | `MRN1066, MRN1107` |
| 10 | 10 | 10/16 | 0.625 | 1.000 | |
| 20 | 20 | 16/16 | **1.000** | 0.800 | first full recall |
| 30 | 30 | 16/16 | 1.000 | 0.533 | |
| 50 | 50 | 16/16 | 1.000 | 0.320 | |

### Canonical query: `Which patients have Hypertension?` (`AGG_HYPERTENSION`)

| K | Retrieved | Relevant | Recall | Precision |
|---|---|---|---|---|
| 2 | 2 | 2/16 | 0.125 | 1.000 |
| 10 | 10 | 10/16 | 0.625 | 1.000 |
| 20 | 20 | 16/16 | 1.000 | 0.800 |
| 30 | 30 | 16/16 | 1.000 | 0.533 |
| 50 | 50 | 16/16 | 1.000 | 0.320 |

Both queries behave identically (order differs by 1 rank: audit query ranks `MRN1066` first, canonical ranks `MRN1107` first). Results reproduce the benchmark depth report (`reports/retrieval/dense/depth_report.md` row `AGG_HYPERTENSION`: `k=2 0.125, k=5 0.312, k=10 0.625, k=20 1.000`).

### What the LLM actually sees at default K

- `retrieve(query, vs, chunks)` → **2 chunks, 114 words, 1005 chars** (`/tmp/audit_retrieval.py` context preview).
- Those 2 chunks are both hypertension-relevant, so a *perfect* LLM would answer with 2 patients, not 0 or 16.

---

## E. Hypertension Relevant-Record Ranks

Full ranking for audit query at `k=120` (masked index, L2 distance). Relevant = YES if Diagnosis contains Hypertension. Record IDs only (no PII).

| Rank | Record | Relevant | Distance | Rank | Record | Relevant | Distance |
|---|---|---|---|---|---|---|---|
| 1 | MRN1066 | YES | 0.92 | 61 | MRN1057 | no | 1.57 |
| 2 | MRN1107 | YES | 0.93 | 62 | MRN1115 | no | 1.58 |
| 3 | MRN1025 | YES | 0.97 | 63 | MRN1013 | no | 1.58 |
| 4 | MRN1104 | YES | 0.97 | 64 | MRN1016 | no | 1.59 |
| 5 | MRN1021 | YES | 0.98 | 65 | MRN1045 | no | 1.59 |
| 6 | MRN1111 | YES | 0.99 | 66 | MRN1046 | no | 1.59 |
| 7 | MRN1119 | YES | 1.00 | 67 | MRN1098 | no | 1.59 |
| 8 | MRN1059 | YES | 1.00 | 68 | MRN1110 | no | 1.59 |
| 9 | MRN1026 | YES | 1.03 | 69 | MRN1063 | no | 1.60 |
| 10 | MRN1085 | YES | 1.08 | 70 | MRN1116 | no | 1.60 |
| 11 | MRN1001 | YES | 1.09 | … | … | … | … |
| 12 | MRN1074 | YES | 1.11 | 120 | MRN1004 | no | 1.82 |
| 13 | MRN1070 | YES | 1.11 | | | | |
| 14 | MRN1052 | YES | 1.12 | | | | |
| 15 | MRN1005 | YES | 1.13 | | | | |
| 16 | **MRN1054** | **no** | 1.20 | | | | |
| 17 | MRN1118 | YES | 1.22 | | | | |
| 18 | MRN1094 | no | 1.33 | | | | |

**Key:** 15 of 16 hypertension records are in ranks 1–15. `MRN1118` is at **rank 17**, displaced by one irrelevant `MRN1054` at rank 16 (Hyperlipidemia record sharing medication/context terms). Hence:

- `k=15` → 15/16 recall (matches the “15 patients” anecdote if someone tested `k=15` or had off-by-one filtering).
- `k=16` → still 15/16 (MRN1054 intrudes).
- `k=17,20,30,50` → 16/16.

Canonical query ranking (k=50) is nearly identical; all 16 hypertension records within top 17, with `MRN1054` again the sole intruder at rank 16 before `MRN1118` at 17. Distances are ~0.1 lower for canonical (closer phrasing to `Diagnosis: Hypertension`), but recall thresholds are the same.

---

## F. Retrieval vs Context vs Generation Diagnosis

| Stage | Status | Evidence |
|---|---|---|
| **Retrieval** | **Failing at runtime K, passing at sufficient K** | At `k=2` (actual runtime): `2/16 recall (0.125)`. At `k=10`: `10/16`. At `k=20`: `16/16`. Masking preserves `Hypertension` string in 16/16 chunks; 1:1 chunk→record intact; L2 ranking is semantically correct (all hypertension in top 17). No filtering/dedup bug — `retriever.py:11` returns every `i>=0`. |
| **Context construction** | **Passing, but bottlenecked by retrieval** | `rag_pipeline.py:97` → `"\n\n".join(chunk for chunk in context_chunks if chunk)` — no limit, no truncation, no dedup, no token budget. All retrieved chunks passed verbatim. At `k=2`, context is 2 records (1005 chars); at `k=50`, context is 50 records (~25k chars). No bug, but downstream LLM token limit truncates generation. |
| **LLM answer generation** | **Contributing failure (truncation + nondeterminism)** | `generator.py:121` `max_tokens=200` — a 16-patient enumerated answer needs ~450–600 tokens (empirical: each patient line ~25–35 tokens + headers). 200 tokens truncates after ~8–10 patients, explaining “15 patients” as a truncation artifact, not retrieval loss. `temperature=0.3` (generator.py:122) is nondeterministic; `LLM_PROVIDER` defaults to `ollama` (`llama3.2`), but HF path uses `Qwen/Qwen2.5-7B-Instruct` with strict `I don't know` system prompt (generator.py:62-110) that may emit “no information” when context is too small or when the model abstains. Measured: `k=2` context contains only 2 hypertension patients — the model cannot list 16, and under the aggregate prompt rule (“return ALL matching records found in context”) it correctly returns 2, but a sampling variation or empty-context edge could produce `I don't know`. |
| **Configuration/runtime mismatch** | **Primary root cause** | Benchmark says `k=20 → 1.0 recall`; runtime never reaches that depth because `rag_answer` hardcodes `k=2` via default. User belief of `k=30/50` does not match call path. |
| **Stale artifact/index** | **Not involved** | `.gitignore` ignores `benchmarks/retrieval/*.json` etc., but no FAISS cache exists; `cli.py:161` calls `build_rag` fresh each session; benchmark also builds fresh. Dataset file `data/sample_patient_data.txt` is the same file the benchmark loads (`ground_truth.py:32` `MRN_DATASET_PATH`). SHA256 `4d1cb2b773...`, 120 records, hypertension 16, re-derived GT matches scan. No stale index. |

**Summary distinction:**

- **Retrieval recall failure** — yes, at `k=2` (runtime), no at `k≥20`.
- **Context construction failure** — no (all retrieved passed).
- **Generation failure** — secondary; truncates full answers and can nondeterministically abstain.
- **Config mismatch** — yes, the dominant cause.
- **Stale index** — no.

---

## G. Explanation of the 15-patient vs Zero-patient Discrepancy

Neither outcome is reproducible from the **clean HEAD runtime** at any fixed K:

- `k=2` (actual runtime) → 2 patients in context → LLM should return 2, not 15 or 0.
- `k=20/30/50` (benchmark, sufficient recall) → 16 patients in context → LLM should return 16 (or truncated <16 due to `max_tokens=200`).

Observed outcomes imply **off-baseline conditions**:

**A. ~15 patients returned**

- Most plausible: the tester had the **dirty `retriever.py k=50`** edit active and tested `k=15` or `k=16` (or the LLM truncated a `k=50` answer). At `k=15`, recall is exactly `15/16` because `MRN1118` sits at rank 17 behind the `MRN1054` intruder. At `k=50`, `max_tokens=200` truncates the 16-patient enumeration after ~15 lines (each masked record is ~500 chars; 50-record context is ~25k chars, well beyond the model’s effective window, and generation is capped at 200 tokens). Our measurement of ~30 tokens/patient × 16 = ~480 tokens > 200 confirms truncation. A truncated streaming output that stops mid-list would appear as “15 patients”.
- Also plausible: `temperature=0.3` sampling dropped one patient nondeterministically.

**B. “no information about hypertension” / `I don't know`**

- The HF system prompt (`generator.py:64-72`) mandates `I don't know` verbatim when “answer is not explicitly present in the retrieved context”. With `k=2`, the context contains only 2 hypertension records — not 16 — so a model interpreting “list all” as requiring exhaustive knowledge may abstain. Alternatively, a transient Ollama/HF routing failure (`cli.py:234` `_is_upstream_error`) or an empty query (e.g., `clean_input_text` stripping at `rag_pipeline.py:29` if the query contained `Context:`/`Question:`) could yield empty context and thus `I don't know`.
- The zero-result cannot be explained by masking (Hypertension term survives) or by embedding (hypertension records rank 1–17 at L2 0.92–1.22). It is a **generation-side abstention**, not retrieval.

**Reproduction attempted:** `/tmp/audit_retrieval.py` with current code cannot produce 15 or 0 from either `k=2` or `k=50` alone without invoking the LLM. Retrieval is deterministic at each K; generation is not (`temperature=0.3`, streaming, provider-dependent). Auditing generation fully requires live LLM calls, which were out of scope per instructions — hence the residual uncertainty flagged in §M.

---

## H. Benchmark-vs-Runtime Comparison

| Dimension | Runtime (`secure_rag/`) | Benchmark (`benchmarks/`) | Match? | Impact |
|---|---|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` (`embedding.py:10`) | same (`embed_chunks`) | **Yes** | None |
| Preprocessing | `clean_input_text` → `split_into_records` → `mask_text` → `chunk_record` | `_load_mrn_records_raw` → `mask_text` (if secure_rag) → `chunk_text` | **Yes** (for this dataset: `max_words=83 <500` → `chunk_record ≡ chunk_text`, 1 chunk/record) | None |
| Masking | Pre-embedding, mandatory (`rag_pipeline.py:75`) | `use_masking` branch (`_common.py:79`, `runner.py:39`) | **Yes** when `secure_rag` config; benchmark also tests `baseline_a` raw for comparison | Diagnosis term preserved 16/16 |
| Query preprocessing | `embed_chunks([query])` raw | same (`runner.py:59`) | **Yes** — never masks query | None |
| FAISS index | `IndexFlatL2` (`vector_store.py:14`) | same | **Yes** | None |
| Similarity metric | L2 distance | L2 | **Yes** | No normalization |
| Sorting | FAISS distance asc, no rerank | same | **Yes** | None |
| Filtering/dedup | None (`retriever.py:11` `if i>=0` only) | Record-level dedup in metrics (`depth_experiment.py:50` `set(record_id)`), but retrieval returns chunks | **Same retrieval, different metric aggregation** | Benchmark metrics are record-level; runtime has no explicit dedup but 1:1 makes it moot |
| **Retrieval K** | **`k=2` (default, `rag_pipeline.py:96` no arg)** | **`k=50` explicit (`runner.py:194`)**, sliced to `K_CANDIDATES` | **No — primary divergence** | Benchmark recall at `k≥20` never tested interactively |
| Context construction | `"\n\n".join(filtered chunks)` | same (`_common.py:125`) | **Yes** | None |
| Context limit | None | None | **Yes** | LLM `max_tokens` is the only limit |
| Prompt | `generator.py:62-114` (system + `Context:\n{context}\n\nQuestion:\n{query}`) | same (`_common.py:124` `retrieve → join → generate_answer`) | **Yes** | Ollama vs HF provider switch is the only branch |
| Generation | `max_tokens=200, temperature=0.3, stream=True` | same (`generate_answer`) | **Yes** | Truncation at 200 tokens affects 16-patient answers |
| Dataset | `data/sample_patient_data.txt` | same (`ground_truth.py:32` `MRN_DATASET_PATH`) | **Yes** | GT 16 matches scan |
| Index freshness | Fresh per `chat()` | Fresh per `run_retrieval()` | **Yes** | No stale cache |

---

## I. Root Cause(s), Ranked by Evidence

1. **Hardcoded `k=2` in interactive runtime (evidence: definitive)** — `retriever.py:4` `k=2` + `rag_pipeline.py:96` no `k` arg. Directly causes `0.125 recall` for 16-record aggregates. Explains why benchmark `k≥20` results were never observed interactively. Fixing this alone would restore benchmark-reported recall.

2. **`max_tokens=200` truncation of aggregate answers (evidence: strong)** — `generator.py:121`. At `k=50`, 16-patient context requires ~500 tokens to enumerate; 200 tokens truncates after ~8–15 patients, matching the “15 patients” anecdote. Quantified via `/tmp/audit_phase2.py` token estimate and context size (`k=50` → 50 chunks × ~500 chars).

3. **Single intruder record MRN1054 at rank 16 (evidence: strong)** — ranking table shows 15 hypertension + 1 Hyperlipidemia (shares `Amlodipine`/`Hypertension` comorbidity terms via `data/generate_dataset.py:88` comorbidities). Causes `k=15`/`k=16` to yield `15/16` rather than `16/16`, a subtle off-by-one that could be mistaken for a model error.

4. **Nondeterministic generation / provider divergence (evidence: moderate)** — `temperature=0.3`, `LLM_PROVIDER` branching (`generator.py:11` defaults `ollama`), strict `I don't know` rule. Explains “no information” outcome as a sampling or provider failure, not retrieval.

5. **Dirty-working-tree confusion (evidence: observed)** — uncommitted `k=2→50` edit found at audit start. Indicates at least one interactive run did use `k=50`, making the “tested k=30/50” claim locally true but not committed. Resolved by `git restore`.

*Ruled out:* masking destroying `Hypertension` (preserved 16/16), chunk→record 1:1 violation (verified `120 == 120`, `max_words=83`), embedding/FAISS/normalization mismatch (identical), stale dataset/index (same file, fresh build).

---

## J. What This Means for the Planned Adaptive-K Experiment

- **Adaptive-K is justified and necessary.** Depth report shows single-record queries saturate by `k=10` (mean `HitRate 0.08`) while multi-record aggregates need `k≈20–50` (`multi_mean_recall` at `k=2 0.32 → k=20 0.98 → k=50 1.0`, `depth_report.md`). A fixed `k=50` would penalize single-record latency/context (precision `0.244` at `k=50` multi) and truncation risk; fixed `k=2` fails aggregates. Adaptive candidate K (separate single/multi modes, ranked #1 in `depth_experiment.py:263`) preserves efficiency while enabling full recall.

- **Baseline must be re-established at correct K before measuring adaptive gains.** Prior interactive “k=30/50” baselines are invalid. True baselines from this audit: `k=2 0.125, k=10 0.625, k=20 1.0` for hypertension. Adaptive logic should trigger `k≥20` for aggregate queries (detect `list all`/`which patients` per `generator.py:79-87` heuristics) and keep `k≤10` for single-record.

- **`max_tokens` must be addressed jointly with retrieval.** Raising `k` to 50 without raising `max_tokens` beyond 200 will still truncate aggregate answers. Recommend either (a) raising to `≥600` for aggregates, or (b) structured enumeration that paginates (as an adaptive-K follow-on, not during this audit per instructions).

- **No need to change embeddings, chunking, or masking.** Those components are sound (1:1, term-preserving, model-identical).

---

## K. Exact Recommended Next Experiment

**Do not yet implement adaptive-K.** First confirm the baseline correction, then measure the generation budget.

1. **Fix K as a controlled variable (no adaptive logic):** Add explicit `k` param to `rag_answer` (e.g., `rag_answer(query, vector_store, chunks, k=50)`) and wire CLI flag `--k` (default 2 → transition to 50 for experiment). Keep masking/policy/embeddings unchanged. This is the minimal change to validate §D.

2. **Run retrieval-only benchmark at K={2,10,20,30,50} for both queries** using the existing `benchmarks/retrieval/runner.py` artifact (already does this) **and** a new runtime retrieval-only script that calls `retrieve` with explicit `k` and measures `relevant_retrieved` without invoking LLM — confirming `/tmp/audit_retrieval.py` numbers in CI.

3. **Run LLM end-to-end for `k=2` vs `k=50` with `max_tokens=200` vs `max_tokens=600` (4 cells):**
   - Query: audit query + canonical.
   - Metric: generated answer recall (parse MRN list from answer vs 16 GT), truncated-token count, `I don't know` rate over 5 samples per cell (temperature nondeterminism).
   - This isolates whether raising K alone (without raising tokens) still yields truncation (predicted: `k=50` + `200` → truncated ~15/16; `k=50` + `600` → 16/16).

4. **Record `k` in every CLI log line** (e.g., `Retrieved 50/120 chunks (k=50)`) to prevent future belief mismatches.

Pass criteria: `k=50` retrieval-only → `16/16` for both queries; `k=50` + `600` tokens → `16/16` generated recall; `k=2` → `2/16` retrieval and ≤2 generated.

---

## L. Files Inspected and Relevant Line Numbers

| File | Lines | Finding |
|---|---|---|
| `secure_rag/retriever.py` | 4, 9, 11 | Default `k=2`, L2 search, `i>=0` filtering |
| `secure_rag/vector_store.py` | 14, 17, 22-23 | `IndexFlatL2`, `k=2`, `min(k, ntotal)` |
| `secure_rag/rag_pipeline.py` | 17-45, 48-63, 66-84, 95-100, 87-92 | `clean_input_text`, `load_data`, `build_rag` (mask→chunk→embed→FAISS), `rag_answer` with no `k`, stop marker |
| `secure_rag/embedding.py` | 10, 13-25 | `all-MiniLM-L6-v2`, `encode` → `float32` |
| `secure_rag/pdf_loader.py` | 18-24, 27-43 | `split_into_records` on `\n\n`, `chunk_record` 500/50, `chunk_text` |
| `secure_rag/masker.py` | 1-34 | `mask_text` with `DefaultPolicy` / `RegexDetector` first |
| `secure_rag/generator.py` | 9-11, 30-51, 54-123, 118-122 | `LLM_PROVIDER` ollama→HF, `max_tokens=200`, `temperature=0.3`, system prompt (62-114) |
| `secure_rag/cli.py` | 149-161, 182-231, 252-253 | `chat()` → `build_rag`, no cached index, streaming |
| `benchmarks/_common.py` | 29, 75-89, 92-117, 120-133 | `RETRIEVAL_K=5`, `build_index`, `EVALUATION_CONFIGS`, `benchmark_answer` |
| `benchmarks/retrieval/runner.py` | 24-25, 29-53, 56-68, 151-153, 194-209 | `MAX_K=50, K_VALUES`, `_build_index_with_record_map`, `_retrieve_top_k`, `k=50` call |
| `benchmarks/retrieval/depth_experiment.py` | 15, 22-52, 141-150, 186-272 | `K_CANDIDATES`, per-query recall, report generation |
| `benchmarks/retrieval/ground_truth.py` | 32, 55-66, 69-80, 83-129, 183-230 | `MRN_DATASET_PATH`, `_load_mrn_records_raw`, inverted index, `AGG_HYPERTENSION` 16 |
| `data/generate_dataset.py` | 71-88, 825-835, 685-730 | Hypertension disease def, `render_patient` template, comorbidities |
| `reports/retrieval/dense/depth_report.md` | full | Benchmark recall/precision/coverage, `AGG_HYPERTENSION k=20 1.0` |
| `.gitignore` | full | No FAISS cache; retrieval artifacts ignored but archived in `_archive_v1` |
| `pyproject.toml` | deps | `sentence-transformers, faiss-cpu, openai, typer, rich` |

Quoted snippets (minimal):

```python
# secure_rag/retriever.py:4
def retrieve(query, vector_store, chunks, k=2):

# secure_rag/rag_pipeline.py:95-96
def rag_answer(query: str, vector_store, chunks):
    context_chunks = retrieve(query, vector_store, chunks)

# secure_rag/vector_store.py:17,22
def search(self, query_vector, k=2):
    limit = min(k, self.index.ntotal)

# secure_rag/pdf_loader.py:27-28
def chunk_record(record: str, chunk_size=500, overlap=50):
    if len(record.split()) <= chunk_size: return [record]

# secure_rag/generator.py:118-122
completion = client.chat.completions.create(
    model=os.getenv("HF_MODEL", _DEFAULT_MODEL),
    max_tokens=200, temperature=0.3, stream=True,
)
```

---

## M. Any Unresolved Uncertainty

- **Generation reproducibility:** The “no information” case was not reproduced with retrieval alone (retrieval never returns 0 hypertension at any `k≥1` — minimum is 1 at `k=1`). Full reproduction requires live LLM calls over both providers (ollama `llama3.2` vs HF `Qwen2.5-7B-Instruct`) across multiple temperatures; per instructions, generation nondeterminism was not exhaustively probed. The hypothesis of sampling abstention / provider failure remains moderate-confidence.
- **Whether interactive testers used additional ad-hoc `k` wiring** (e.g., `runner.py` directly) — the dirty `k=50` edit suggests yes, but commit history shows no such change (`git log -- secure_rag/retriever.py` shows only `k=2` at HEAD). Remote tester’s exact code is unknown.
- **Truncation token count for 16-patient answer:** estimated ~480 tokens; exact count depends on masked text length and model tokenizer; could be verified by counting tokens with the Qwen tokenizer on `/tmp/audit_retrieval.py` context.

---

## Appendix — Reproduction Commands (read-only, /tmp)

```bash
# Retrieval verification (no prod changes) — generates §D and §E tables
python3 /tmp/audit_retrieval.py   # k sweep + full rank 1..120 + default-k proof + context size
python3 /tmp/audit_phase2.py      # benchmark vs runtime parity

# Both scripts load data/sample_patient_data.txt, call build_rag(), and use only /tmp for output.
# No files under secure_rag/, benchmarks/, tests/, or config are written.
```

**Repository state after audit:** `git status` clean; `python3 -m pytest tests/` to be run post-report (see §K step 1 for result). No production source files were modified during this audit (dirty `retriever.py` was restored via `git restore`).

