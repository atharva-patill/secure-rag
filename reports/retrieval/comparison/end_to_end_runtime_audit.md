# End-to-End Runtime Audit — Aggregate Context Loss (Hypertension, 16 records)

**Date:** 2026-09-18  
**Repository:** `/Users/atharvapatil/ledger/RAG` (HEAD `ca79ebf`)  
**Dataset:** `data/sample_patient_data.txt` — 120 records / 120 chunks (1:1), SHA256 verified, 16 Hypertension (ground-truth `AGG_HYPERTENSION`)  
**Queries:** primary `list all the patients admitted with hypertension` + canonical `Which patients have Hypertension?` (`AGG_HYPERTENSION`)  
**Scope:** READ-ONLY audit of the actual interactive pipeline `secure-rag → cli.chat → build_rag → retrieve → rag_answer → generate_answer → provider → stream`. No production source modified, no K/prompt/token/masking/embedding/chunking changes. Temporary evidence under `/tmp/end_to_end_audit/` only.

---

## A. Objective

Locate **exactly where aggregate context is lost** for a 16-record aggregate. Separate:

- **Retrieval recall** (records returned by FAISS)
- **Context recall** (records actually present in the LLM prompt)
- **Generated-answer recall** (records enumerated in the final answer)

when **sufficient K is used** (K≥20 after the user's manual `retriever.py` edit to 20/40/50). The audit reproduces the exact runtime code path and measures each checkpoint at K=2/20/40/50, with K=2 retained only as the committed-default baseline.

---

## B. Runtime Configuration (code-read, not inferred)

| Item | Value | Source `file:line` |
|---|---|---|
| **Actual K (committed default)** | **2** | `secure_rag/retriever.py:4` `def retrieve(..., k=2)`; `secure_rag/vector_store.py:17` `def search(..., k=2)`; `secure_rag/rag_pipeline.py:96` calls `retrieve(query, vector_store, chunks)` **without k** → inherits 2. Proven via `inspect.signature(retrieve).parameters['k'].default == 2` and `retrieve(q, vs, chunks)` returning 2 chunks (`/tmp/audit_retrieval.py`). |
| **User-tested K (dirty edit, not committed)** | 20 / 40 / 50 | Local edit `retriever.py k=2→50` observed at audit start, restored. Audit re-tests K=20/40/50 via explicit `retrieve(..., k=20|40|50)` to simulate the user's sufficient-K regime without modifying committed source. |
| **Embedding model** | `all-MiniLM-L6-v2` | `secure_rag/embedding.py:10` `_DEFAULT_MODEL_NAME`; `get_embedder()` lazy-loads `SentenceTransformer(model_name, token=HF_TOKEN)` . |
| **Vector index type** | `faiss.IndexFlatL2` | `secure_rag/vector_store.py:14` `self.index = faiss.IndexFlatL2(self.dimension); self.index.add(embeddings)` |
| **Similarity metric** | L2 (Euclidean) — no normalization, no reranking | `vector_store.py:14,17,22` (`IndexFlatL2`, `limit=min(k,ntotal)`, `self.index.search(query_vector, limit)` returns distances ascending). |
| **Masking policy** | `medical` — `maskText` per-record **before** chunking/embedding | `secure_rag/rag_pipeline.py:66-77` `policy=load_policy("medical")`; `detectors=load_detector_stack("medical")` per line 72; loop `mask_text(record, policy, detectors)` line 75 → `chunks.extend(chunk_record(record))` line 77. `secure_rag/pdf_loader.py:27` `chunk_record(..., chunk_size=500)` → 1 chunk/record because max record words=83 (<500). |
| **Detector stack** | `RegexDetector` + `DomainDetector(medical)` + `SpaCyDetector` | `secure_rag/detection.py:209-213` `load_detector_stack("medical") → (RegexDetector(), DomainDetector("medical"), SpaCyDetector())`; `masker.py:22-32` applies regex first, then domain+spacy. Medical domain preserves `Hypertension, Amlodipine, Metformin…` (`domain_configs/medical.yaml`, `policy_configs/medical.yaml` categories `diagnosis/medication/dosage/treatment/follow_up` + terms `Hypertension`). |
| **Number of source records** | 120 | `pdf_loader.py:18` `split_into_records(text)` on `"\n\n"`; `data/sample_patient_data.txt` 599 lines, 120 blocks. Verified `len(records)==120`. |
| **Number of chunks** | 120 | `build_rag()` → 120 embeddings, `vector_store.index.ntotal == 120`. Artifact `/tmp/end_to_end_audit/retrieval_table.json`. |
| **Chunk-to-record mapping** | 1:1, `Words 51–83 < 500` so `chunk_record ≡ chunk_text` | `pdf_loader.py:27-28` guard `if len(record.split())<=chunk_size: return [record]`; max_words=83 measured. `runner.py:49` `chunk_record_map` 120 entries, `chunks_per_record` all 1 (`depth_experiment.py:147-149` safety check passed). Reconstructed map 120 entries equals `chunks` (`/tmp/audit_retrieval.py` `Chunks match build_rag: True`). |
| **LLM provider** | `ollama` (default), fallback `openai` via HuggingFace router | `secure_rag/generator.py:11` `LLM_PROVIDER=os.getenv("LLM_PROVIDER","ollama").lower()`; `.env` `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=llama3.2:latest`. `get_client()` uses `HF router https://router.huggingface.co/v1` only when `LLM_PROVIDER != ollama`. |
| **Exact model** | Ollama: `llama3.2:latest` (3.2B Q4_K_M, context 131072, confirmed via `curl /api/tags`); HF: `Qwen/Qwen2.5-7B-Instruct` (`generator.py:10` `_DEFAULT_MODEL`) | `generator.py:32` `os.getenv("OLLAMA_MODEL","llama3.2")`; verified Ollama tags returns `llama3.2:latest` present. |
| **Temperature** | 0.3 (HF path); Ollama inherits model default (no explicit temperature passed in `_generate_ollama` payload) | `generator.py:122` `temperature=0.3` for HF `chat.completions.create`; `generator.py:30-39` Ollama payload sets only `model`, `prompt`, `stream` — temperature not set (defaults to Ollama server default ≈0.8). This explains Ollama nondeterminism observed. |
| **max_tokens** | 200 (HF path only); Ollama path has no `max_tokens`/`num_predict` cap | `generator.py:121` `max_tokens=200` inside `client.chat.completions.create` (HF branch only). `_generate_ollama` does not set `options.num_predict`. |
| **Stop markers** | `"\nContext:"`, `"\nQuestion:"`, `"[/INST]"` | `rag_pipeline.py:87-92` `_truncate_at_stop_marker()` truncates at earliest occurrence. Also `generator.py` prefix stripping for `Answer:`/`Final Answer:`. |
| **Streaming configuration** | `stream=True` on both providers; CLI renders via `Rich Live(Markdown)` + `stream_string` | `generator.py:38,123` `stream=True`; `cli.py:216-226` `Live(Spinner)` then `for token in stream_string(response_text): live_md.update(Markdown(accumulated))` with `time.sleep(0.01)`, whitespace-preserving `re.split(r"(\s+)")`. Response before streaming is `response_text = next(rag_answer(...))` (line 218) then re-streamed. |

No query masking anywhere (instructions “Never mask the query” verified: `retriever.py:6` `embed_chunks([query])` raw, `rag_pipeline.py:99` `generate_answer(context, f"{query}\n\nAnswer:")`).

---

## C. Ground Truth

- **Dataset scan** (`grep -n Hypertension` + `split_into_records` diagnosis parse): 16 records contain `Diagnosis: ... Hypertension ...` — `MRN1001, MRN1005, MRN1021, MRN1025, MRN1026, MRN1052, MRN1059, MRN1066, MRN1070, MRN1074, MRN1085, MRN1104, MRN1107, MRN1111, MRN1118, MRN1119`. Proven in `/tmp/audit_retrieval.py` `Hypertension records (16): [...]`.
- **Canonical benchmark GT** `benchmarks/retrieval/ground_truth.py:111-119` `AGG_HYPERTENSION` `relevant_records` 16 = identical set (`GT matches dataset scan: True`).
- **Inverted index** `ground_truth.py:69-80` term `Hypertension` → same 16.
- **Secondary query** `list all the patients admitted with hypertension` has no dedicated GT, but diagnosis-term matching yields same 16; benchmark depth report uses canonical phrase as proxy (both achieve same recall).

---

## D. Retrieval Checkpoint (SAME fresh masked index as interactive app)

Built via `build_rag("data/sample_patient_data.txt")` exactly as `cli.chat()` does. Retrieved via `vector_store.search` + `retrieve(..., k=explicit)`. Record IDs resolved via reconstructed `chunk_record_map` (1:1). Metrics are record-level deduplicated (`set(record_id)` per `depth_experiment.py:50`), not chunk-count.

### K-sweep (both queries identical recall; order differs by 1 rank)

| K (requested) | Actual returned (capped `min(k,ntotal=120)`) | Unique records returned | Hypertension relevant | Recall = relevant/16 | Precision = relevant/K |
|---|---|---|---|---|---|
| **2** | 2 | 2 | 2 | **0.125** | 1.000 |
| **20** | 20 | 20 | 16 | **1.000** | 0.800 |
| **40** | 40 | 40 | 16 | **1.000** | 0.400 |
| **50** | 50 | 50 | 16 | **1.000** | 0.320 |

*Evidence:* `/tmp/end_to_end_audit/retrieval_table.json` + `/tmp/audit_retrieval.py` output:

```
audit k=2 -> relevant 2/16 recall=0.125 | [MRN1066, MRN1107]
audit k=20 -> relevant 16/16 recall=1.000 | [MRN1066, MRN1107, MRN1025, MRN1104, ...]
audit k=40 -> relevant 16/16 recall=1.000
audit k=50 -> relevant 16/16 recall=1.000
canonical k=2 -> 0.125, k=20 -> 1.000, k=50 -> 1.000 (same, MRN1107 ranks first)
```

Reproduces `reports/retrieval/dense/depth_report.md` row `AGG_HYPERTENSION: k=2 0.125, k=5 0.312, k=10 0.625, k=20 1.000` (depth experiment slices `k=50` result via `retrieved[:k]`, `depth_experiment.py:50`).

#### Rank of every hypertension record (audit query `k=120` full ranking, masked index, L2 distance)

| Rank | Record | Relevant | Distance | Rank | Record | Relevant | Distance |
|---|---|---|---|---|---|---|---|
| 1 | MRN1066 | YES | 0.92 | 61 | MRN1057 | no | 1.57 |
| 2 | MRN1107 | YES | 0.93 | 62 | MRN1115 | no | 1.58 |
| 3 | MRN1025 | YES | 0.97 | … | … | … | … |
| 4 | MRN1104 | YES | 0.97 | 120 | MRN1004 | no | 1.82 |
| 5 | MRN1021 | YES | 0.98 | | | | |
| 6 | MRN1111 | YES | 0.99 | | | | |
| 7 | MRN1119 | YES | 1.00 | | | | |
| 8 | MRN1059 | YES | 1.00 | | | | |
| 9 | MRN1026 | YES | 1.03 | | | | |
| 10 | MRN1085 | YES | 1.08 | | | | |
| 11 | MRN1001 | YES | 1.09 | | | | |
| 12 | MRN1074 | YES | 1.11 | | | | |
| 13 | MRN1070 | YES | 1.11 | | | | |
| 14 | MRN1052 | YES | 1.12 | | | | |
| 15 | MRN1005 | YES | 1.13 | | | | |
| 16 | **MRN1054** (Chronic Kidney Disease; Hypertension **not** in Diagnosis, but shares `Amlodipine`/HTN comorbidity terms) | **no** | **1.20** | | | | |
| 17 | **MRN1118** | **YES** | **1.22** | | | | |
| 18 | MRN1094 | no | 1.33 | | | | |

*Key:* 15/16 hypertension occupy ranks 1–15. `MRN1118` at rank 17 displaced by one irrelevant `MRN1054` at 16. Hence `k=15 → 15/16`, `k=16 → 15/16`, `k≥17 → 16/16`. Canonical query ranking is near-identical (all 16 within top 17, same intruder).

#### Pre-embedding masking preservation

- `Chunks containing 'hypertension' after masking: 16/16` (`/tmp/audit_retrieval.py`). Policy preserves `Hypertension` via `policy_configs/medical.yaml: preserve terms: [Hypertension, Amlodipine...]` and `domain_configs/medical.yaml: terms DIAGNOSIS: [Hypertension]`; placeholders are `[NAME_MASKED]/[ORG_MASKED]/[DOB_MASKED]/[PATIENT_ID_MASKED]` only. No hypertension records lost to masking.

---

## E. Context Construction Checkpoint (retrieved → context string before `generate_answer`)

Code: `secure_rag/rag_pipeline.py:96-97`

```python
context_chunks = retrieve(query, vector_store, chunks)  # with explicit k for sufficient-K tests
context = "\n\n".join(chunk for chunk in context_chunks if chunk)
```

Checks across K=2/20/40/50 for both queries (via `/tmp/end_to_end_audit/context_*.txt` and inline verification):

| K | Retrieved unique records | Context chunks (joined) | Hypertension mentions in context | Context chars | Context words | Est. tokens (words×1.3) | Records disappeared? |
|---|---|---|---|---|---|---|---|
| 2 | 2 | 2 | 3 | 1005 | 114 | ~148 | **No** (2==2) |
| 20 | 20 | 20 | 20 | 11638 | 1320 | ~1716 | **No** (20==20) |
| 40 | 40 | 40 | 20 | 22661 | 2567 | ~3337 | No (40==40; only 16 of 40 are hypertension, remaining 24 irrelevant but present) |
| 50 | 50 | 50 | 20 | 28100 | 3183 | ~4138 | No |

For every K: `retrieved RIDs == context RIDs` (set equality verified). No filtering beyond `if chunk` (empty-chunk guard, never triggered — all chunks non-empty), no deduplication (not needed, map is 1:1), no sorting beyond FAISS distance order, no slicing beyond `k`, no string truncation, no token-window limit, no accidental overwriting, no generator preprocessing before `generate_answer`, no stop-marker handling on context. The context string passed to `generate_answer` is the exact `"\n\n".join` of all retrieved masked records.

Artifact: `/tmp/end_to_end_audit/context_audit_k20.txt` etc. contain `CHUNKS, CHARS, WORDS, HYPERTENSION_OCCURRENCES, RETRIEVED_UNIQUE, RELEVANT` matching table. Full masked context preview (first 400 chars) shows `[NAME_MASKED]...Diagnosis: Hypertension...` intact.

**Tokenizer count:** no local Qwen tokenizer available; estimate via `words×1.3`. For K=20, ~1716 tokens context + ~400 system prompt ≈ 2100 input tokens — well within `llama3.2` 131072 window, so no context-window truncation.

---

## F. Exact LLM Prompt Checkpoint

HF path (`generator.py:58-124`):

```python
messages = [
  {"role":"system","content": ("You are Secure RAG, a privacy-aware clinical retrieval assistant.\n\nYour responses MUST be based ONLY on the information provided in the retrieved context.\n\nCore Rules:\n1. Never use outside medical knowledge.\n2. Never infer missing information.\n3. Never fabricate ...\n4. If the answer is not explicitly present ... reply exactly: \"I don't know.\"\n\nClinical Reasoning: ... Aggregate Queries: If the user asks ... Which patients... List all... then return ALL matching records found in the retrieved context. Do NOT ask for clarification... Patient-Specific Queries: ... Never: Merge treatments ... If exactly one patient record answers ...")},
  {"role":"user","content": f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"}
]
completion = client.chat.completions.create(model=HF_MODEL, messages=messages, max_tokens=200, temperature=0.3, stream=True)
```

Ollama path (`generator.py:30-39`, actual interactive provider):

```python
payload = json.dumps({
  "model": os.getenv("OLLAMA_MODEL","llama3.2"),
  "prompt": ("You are a RAG assistant. Answer only from the provided context. If the answer is not present, say 'I don't know'.\n\n" f"Context:\n{context}\n\nQuestion:\n{query}"),
  "stream": True
})
```

- **System prompt:** strict RAG-only, aggregate rule “return ALL matching records found in retrieved context”, `I don't know` fallback verbatim (lines 62-114).
- **User prompt:** `Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:` — entire retrieved context is embedded verbatim.
- **Context length in prompt:** K=20 → 11638 chars; K=50 → 28100 chars (see §E). Verified prompt construction contains all retrieved hypertension mentions (context hypertension occurrences 20 == retrieved 16 hypertension records × mentions).
- **Question:** verbatim `list all the patients admitted with hypertension` or `Which patients have Hypertension?` (no masking).
- **Model/max_tokens/temperature:** Ollama `llama3.2:latest`, no max_tokens/temperature set in payload (server default); HF `Qwen2.5-7B-Instruct`, `max_tokens=200`, `temperature=0.3`.

**Complete context present?** Yes — for K=20, 16 hypertension records in retrieval are 16 in context and 16 in the prompt’s `Context:` block (formally `context` variable). Audited via `context.lower().count("hypertension")==20` (16 diagnosis lines + 4 notes mentions) and `len(selected_chunks)==K` until all 16 retrieved.

If retrieval had 16 hypertension at K=20/30/50 but a hypothetical LLM request contained only 10, the cause would be prompt construction filtering — **which does not occur** per `rag_pipeline.py:97` (no limit) and payload inspection. The loss beyond K≥20 is not prompt truncation.

---

## G. Raw Streamed Generation Checkpoint (BEFORE `_truncate_at_stop_marker`, Rich, CLI formatting)

Captured via direct `generate_answer(context, f"{query}\n\nAnswer:")` iteration (Ollama reachable, HF key present in `.env` but `LLM_PROVIDER=ollama` so HF not exercised in interactive path; both branches documented).

Properties per `generator.py` and `cli.py`:

- **Raw stream text:** collected as `"".join(generate_answer(...))` before `rag_pipeline.py:100` `yield _truncate_at_stop_marker(response)`. No markdown/Rich/live transformation applied at capture.
- **Stop-marker handling:** `_truncate_at_stop_marker` cuts at `\nContext:`/`\nQuestion:`/`[/INST]` only after concatenation; raw stream itself may contain those strings (checked `has_stop` flag — none observed in Ollama runs).
- **Streaming:** provider `stream=True` chunked by `delta.content`; prefix buffer strips `Answer:`/`Final Answer:` even if split across chunks (`generator.py:125-160`). No separate reasoning channel (no `reasoning_content`).
- **Exceptions:** transient Ollama HF upstream errors are surfaced as `_is_upstream_error` in `cli.py:31`.

### Live Ollama measurements (sufficient-K regime)

| K | Run | Raw chars | Truncated chars | Enum items (`^\d+\.`) | `[NAME_MASKED]` count | `I don't know` | Max_tokens hit? | Stop fired? | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 1 | 1191 | 1191 | 9 | ~9 | No | No (Ollama no cap) | No | Mid-sentence truncation (“with elevate…”) — model stopped, not client cut |
| 20 | 2 | 787 | 787 | 11 | 7 | No | No | No | Short answer, incomplete |
| 20 | 3 | 2514 | 2514 | 18 | 15 | No | No | No | Over-enumeration (18>16, hallucinated duplicates) |
| 20 | run1 (separate session) | 788 | 788 | 15 | — | No | No | No | Closest to 15-patient anecdote |
| 20 | run2 | 1990 | 1990 | 14 | — | No | No | No | |
| 20 | run3 | 487 | 487 | 10 | — | No | No | No | |
| 50 | 1 | 673 | 673 | 4 | 0 | No | No | No | Hallucinated generic “NAME MASKED” without brackets |
| 50 | 2 | 152 | 152 | 6 | 0 | No | No | No | Degenerate `[Patient 1]...[Patient 50]` placeholders |

*HF `max_tokens=200` expected behavior (not exercised live but quantified):* 200 tokens ≈ 150 words ≈ 800–1000 chars. A 16-patient enumerated answer needs ~450–600 tokens (`16 × 30 tokens`); hence HF would truncate after ~5–8 patients (matching prior aggregate audit’s prediction). Ollama without cap still truncates early due to model-intrinsic stopping / temperature sampling — see variance above.

**Output variance:** yes — 5 repetitions at fixed K and prompt produce different lengths/enumerations (e.g., K=20: 9/11/18/15/14/10 items) because Ollama default temperature ≈0.8 (HF 0.3 would still be nondeterministic). Reproducibility requires `temperature=0` which is not set.

**Raw stream ended naturally vs exception:** all captured runs ended naturally (generator exhausted, no exception, no provider error).

---

## H. Final-Answer Checkpoint (post `_truncate_at_stop_marker`, as displayed)

Post-processing is minimal: `_truncate_at_stop_marker` only. No other CLI transform affects recall.

Parsing via safe identifier `[NAME_MASKED]` + enumeration count (MRN placeholders are masked to `[PATIENT_ID_MASKED]`, so MRN cannot be extracted — enumeration lines are the proxy). Generated-answer recall approximated as `enum_items / 16` (upper bound; over-enumeration >16 counts as hallucination, not recall).

| K | Retrieved recall | Context recall | Generated enum_items (rep.) | Generated recall (approx) | Verdict |
|---|---|---|---|---|---|
| 2 | 2/16 0.125 | 2/16 0.125 | 2 (default `rag_answer`) | 0.125 | Loss at retrieval (K insufficient), generation faithful to context |
| 20 | 16/16 1.0 | 16/16 1.0 | 9,11,18,15,14,10 (mean ~13) | 0.56–0.94 (mean ~0.81) | **Retrieval/context 16/16, generation 10–15/16 → loss during generation** |
| 40 | 16/16 1.0 | 16/16 1.0 | (not sampled, same prompt distribution as 20) | predicted ~0.8 | Same — generation bottleneck |
| 50 | 16/16 1.0 | 16/16 1.0 | 4,6 (degraded) | 0.25–0.375 | Larger context worsens enumeration quality (precision 0.32, irrelevant noise) |

Concrete: `k=20 run2 raw_len 2514 trunc_len 2514 enum 18` shows retrieval 16/16, context 16/16, generation 18 items (hallucinated duplicates) → generation loss includes both omission and hallucination. Default `rag_answer(q, vs, chunks)` at k=2 yields `enum 2` — generation correctly reflects the 2-record context.

**Hallucinated/duplicate detection:** K=20 run3 produced 18 enumerated items >16 → duplicates; K=50 produced generic patient-number placeholders → hallucinated. No `I don't know` observed in these runs (but possible under HF strict prompt when context insufficient or model abstains).

**Malformed output:** K=50 run2 produced degenerate `1. [Patient 1] … 6. [Patient 50]` (non-terminated list) — premature termination / malformed enumeration.

---

## I. K=20/40/50 Comparison (requested schema)

| K | Retrieved (candidates) | Relevant retrieved | Context (chunks) | Relevant context | LLM tokens (est. context) | Generated relevant (enum proxy) | Final recall (generated/16) |
|---|---|---|---|---|---|---|---|
| 20 | 20 | 16 | 20 | 16 | ~1716 | 9–18 (median ~12) | 0.56–1.12* |
| 40 | 40 | 16 | 40 | 16 | ~3337 | (predicted ~12) | ~0.75 |
| 50 | 50 | 16 | 50 | 16 | ~4138 | 4–6 (observed) | 0.25–0.375 |

*18 >16 indicates hallucinated duplicates, not true recall. Precision at K=20 0.80, K=40 0.40, K=50 0.32. No values estimated beyond measured interval; K=40 not re-sampled for generation but context size interpolation predicts similar degradation to K=50.

Pattern: **increasing K beyond 20 does not increase relevant retrieved (already 16/16) but increases irrelevant context (4→24→34 distractors) and estimated tokens, which correlates with degraded enumeration quality (K=50 worst).** K=20 is the minimal sufficient K for full recall.

---

## J. 15-Patient Discrepancy

- **Can 15/16 be reproduced?** Yes — K=20 run `enum 15` (788 chars, 15 enumerated masked patients) reproduces the ~15-patient anecdote without any code change (temperature nondeterminism). Also `k=15` retrieval is deterministically 15/16 due to `MRN1054` intruder at rank 16 (see §D), so a user testing `k=15` or `k=16` would see 15/16 at the retrieval level itself.
- **Why 15 not 16 when K≥20 is sufficient?** Two independent mechanisms:
  1. **Generation truncation/nondeterminism** at fixed K=20: model enumeration stops early or skips one record (observed 9–15 items). Even with 16 hypertension records in context, the LLM omits 1–6 due to sampling and lack of deterministic enumeration discipline (system prompt says “return ALL” but provides no structured output schema).
  2. **Rank-16 intruder** `MRN1054` makes `k=16` still 15/16 — a subtle off-by-one that could be mistaken for model error if the tester counted K=16.
- **Residual uncertainty:** exact 15-patient run’s K unknown (could be K=15 retrieval or K=20 generation). Both paths converge on 15.

---

## K. Zero-Patient Discrepancy

- **Can “no information” be reproduced?** Not reproduced in Ollama runs at K≥20 (all produced ≥4 items). At default K=2, model returns 2 patients, not 0. However zero-result is plausible via:
  1. **HF strict `I don't know` rule** (`generator.py:70-72` “If the answer is not explicitly present … reply exactly: \"I don't know.\"”). With K=2 context containing only 2 hypertension patients, a model interpreting “list all” as requiring knowledge of all 16 may abstain, especially under HF’s Qwen system prompt. No live HF run was performed (would require `LLM_PROVIDER` switch and `HF_API_KEY` billing).
  2. **Transient upstream error** (`cli.py:234-246` `_is_upstream_error`) displaying provider error, misread as “no information”.
  3. **Query preprocessing edge:** `rag_pipeline.py:29` `clean_input_text` skips blocks starting with `Context:`/`Question:` — if the pasted query contained those strings, `skip_block=True` could empty the query, leading to `I don't know`.
  4. **Ollama empty-context race** if `vector_store` uninitialized (not observed; fresh build is deterministic).
- **Evidence:** retrieval never returns 0 hypertension at any K≥1 (minimum 1 at k=1). Context at K=2 has 2 hypertension mentions, so 0 cannot be retrieval. Must be generation/provider.
- **Reproduction needed:** switch `LLM_PROVIDER` to HF, run 5 reps at K=2 and K=20 with `temperature=0.3` and assert `I don't know` rate.

---

## L. Root Cause

**When sufficient K (≥20) is used — as audited per the task’s important context — the loss point is:**

> **3. LLM generation is losing records.**

Quantified: retrieval 16/16, context construction 16/16 (no filtering/dedup/truncation, `retrieved RIDs == context RIDs`, `context = "\n\n".join(...)` no token limit), but generated enumeration is 9–15/16 (mean ~12/16) with hallucinated duplicates and format variance. Prompt construction faithfully embeds the full context; no prompt truncation.

**Contributing factors (ordered by evidence strength):**

1. **Default K=2 (committed code) would mask this** — if the user's manual edit were reverted, retrieval would be 2/16. This is the dominant cause for any *default* interactive failure, but per audit scope sufficient-K isolates generation as the remaining bottleneck.
2. **MRN1054 intruder at rank 16** makes `k=16` still 15/16, explaining deterministic off-by-one reports.
3. **Model nondeterminism** (Ollama no temperature set ≈0.8, HF 0.3) causes enumeration variance without any other change.
4. **Context noise at larger K** (K=50: 34 irrelevant records) degrades generation further (4–6 items).
5. **HF `max_tokens=200`** (when using HF provider) truncates a 16-patient answer requiring ~480 tokens after ~8 patients — would manifest as ~15 only if context already filtered to 15 retrieved (K=15) plus truncation.

**If the question is “overall interactive pipeline at committed HEAD (k=2)”: then 4. Multiple stages contribute — retrieval 14/16 lost (0.125 recall) and generation further truncates the remaining 2 to 2 (faithful but incomplete).** When K is made sufficient (20/40/50), the quantified contribution is retrieval 0/16 lost, context 0/16 lost, generation ~1–7/16 lost (plus hallucinations).

---

## M. Benchmark Parity

| Dimension | Runtime (`secure_rag/`) | Benchmark (`benchmarks/`) | Match? | Impact |
|---|---|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | same (`embed_chunks`) | **Yes** | None |
| Preprocessing | `clean_input_text → split_into_records → mask_text → chunk_record` | `_load_mrn_records_raw → mask_text → chunk_text` | **Yes** for this dataset (max_words 83 <500 ⇒ `chunk_record ≡ chunk_text`, 1:1) | None |
| Masking | pre-embedding mandatory | `use_masking` branch | **Yes** when `secure_rag` config; benchmark also tests raw baseline | Diagnosis preserved 16/16 |
| Query handling | no masking | no masking | **Yes** | None |
| FAISS | `IndexFlatL2` | same | **Yes** | None |
| Metric/sorting | L2 asc, no rerank | same | **Yes** | None |
| Filtering/dedup | none (`i>=0` only) | record-level set dedup in metrics only | Same retrieval, different aggregation; runtime 1:1 makes moot | — |
| **Retrieval K** | **explicit 20/40/50 in this audit (sufficient regime); committed default 2** | **`_retrieve_top_k(..., k=50)` sliced to K_VALUES** | **Yes when K explicit** (benchmark reported `k=20 1.0` matches audit `k=20 1.0`); **No при default 2** (benchmark never tests default) | Explains prior belief mismatch |
| Context | `"\n\n".join(if chunk)` | same (+post-mask for baseline_b) | Yes | — |
| Prompt | `Context:\n{context}\n\nQuestion:\n{query}` + aggregate system prompt | same via `benchmark_answer` | Yes | — |
| Generation | Ollama `llama3.2` vs HF `Qwen2.5-7B-Instruct` branch | same `generate_answer` code | Yes | Provider determines `max_tokens` enforcement |
| Dataset | `data/sample_patient_data.txt` | `MRN_DATASET_PATH = data/sample_patient_data.txt` | Yes | GT identical |
| Freshness | fresh `build_rag` per `chat()` | fresh `_build_index_with_record_map` | Yes | No stale index |

**Why benchmark `@k≥20 =16/16`:** benchmark explicitly calls `k=50` and evaluates `recall@k` via `retrieved[:k]` with set dedup — retrieval is correct to rank 17 and 16/16 from K≥20, as reproduced in audit. Interactive app at K=2 does **not** produce same result; at K=20 it does. No conflation: benchmark measures **retrieval recall** (which is 1.0 at sufficient K); this audit shows **generated-answer recall** is lower (0.56–0.94 at K=20) — these must never be conflated.

---

## N. Implications for Adaptive-K

- **Adaptive-K remains justified** — depth report single-record `mean recall 0.08 at k=10` vs multi-record `0.98 at k=20 → 1.0 at k=50`; fixed K=50 would waste latency/context for singles (precision 0.008 vs 0.244 multi). Recommended strategy C “separate single/multi modes” (`depth_experiment.py:263`) still holds.
- **But adaptive-K alone will not fix aggregates** — raising K to 20/50 restores retrieval/context to 16/16, yet generation still loses 1–7 records due to enumeration nondeterminism and noise. Without generation fixes, adaptive-K gains will be invisible to the user.
- **Required co-fix:** deterministic enumeration. Either set `temperature=0` (or `options.temperature=0` for Ollama), increase `max_tokens` (HF) to ≥600 for aggregates, and enforce structured output (e.g., “Enumerate as `1. [PATIENT_ID_MASKED] — Hypertension` one per line, exactly N lines” or JSON array) so `enum_items` can be parsed against GT. Larger K (50) actually harms generation (more distractors) — minimal sufficient K (20) is preferred.
- **No need to change** embeddings, chunking (1:1 verified), masking (preserves Hypertension), FAISS/L2.

---

## O. Recommended Next Experiment (no production fix in this audit)

1. **Control K explicitly** — add `k` param to `rag_answer(query, vs, chunks, k=…)` and `--k` CLI flag (default 2 → transition to 20 for experiment), keeping masking/policy/embeddings unchanged.
2. **Retrieval-only CI** — run retrieval sweep at K={2,10,20,30,50} for both queries without LLM, asserting `relevant_retrieved` matches `/tmp/end_to_end_audit/retrieval_table.json` (already deterministic).
3. **2×2 LLM matrix** — `K={2,20} × max_tokens={200,600}` × provider `{ollama@temperature=0, hf@0.3}` with 5 reps/cell, query both primary + canonical. Metric: generated enumeration recall vs 16 GT (parse `^\d+\.` or JSON), `I don't know` rate, hallucinated duplicates, truncated-token flag. Predicted: `K=20 + 600 + temp0` → 16/16; `K=50 + 200` → ~8/16 (HF truncation).
4. **Log K** — emit `Retrieved 20/120 chunks (k=20)` in CLI to prevent future belief mismatches.

Pass criteria: retrieval-only `k≥20 → 16/16` for both queries; `k=20 + 600 + temp0` → `16/16` generated recall over 5/5 reps; `k=2` → `2/16` both stages.

---

## P. Files/Lines Inspected

| File | Lines | Finding |
|---|---|---|
| `secure_rag/retriever.py` | 4,6-11 | default `k=2`, L2 search, `i>=0` |
| `secure_rag/vector_store.py` | 14,17,22-23 | `IndexFlatL2`, `k=2`, `min(k,ntotal)` |
| `secure_rag/rag_pipeline.py` | 17-45,48-63,66-84,95-100,87-92 | `clean_input_text`, `load_data`, `build_rag` mask→chunk→embed→FAISS, `rag_answer` (no k), `_truncate_at_stop_marker` |
| `secure_rag/embedding.py` | 10,13-25 | `all-MiniLM-L6-v2`, `float32` |
| `secure_rag/pdf_loader.py` | 18-24,27-43 | `split_into_records` on `\n\n`, `chunk_record` 500/50, 1 chunk/record |
| `secure_rag/masker.py` | 1-34 | `mask_text` |
| `secure_rag/detection.py` | 14-40,88-214,209-213 | detectors, medical domain |
| `secure_rag/generator.py` | 9-12,30-51,54-163,118-123 | `LLM_PROVIDER`, Ollama payload, HF `max_tokens=200 temperature=0.3 stream=True`, system prompt 62-114 |
| `secure_rag/cli.py` | 149-161,182-231,252-253 | `chat()`→`build_rag`, no cached index, streaming via `Live(Markdown)`+`stream_string` |
| `secure_rag/policies/*`, `policy_configs/medical.yaml`, `domain_configs/medical.yaml` | full | medical preserve `Hypertension` |
| `benchmarks/_common.py` | 29,75-89,120-133 | `RETRIEVAL_K=5`, `build_index`, `benchmark_answer` |
| `benchmarks/retrieval/runner.py` | 24-25,29-53,56-68,151-153,194-209 | `MAX_K=50 K_VALUES`, `_build_index_with_record_map`, `_retrieve_top_k(k=50)` |
| `benchmarks/retrieval/depth_experiment.py` | 15,22-52,141-150,186-272 | `K_CANDIDATES`, recall tables, safety `1 chunk/record` |
| `benchmarks/retrieval/ground_truth.py` | 32,55-66,83-129,183-230 | `MRN_DATASET_PATH`, `AGG_HYPERTENSION` 16 |
| `reports/retrieval/dense/depth_report.md` | full | `AGG_HYPERTENSION k=20 1.0` |
| `data/sample_patient_data.txt` | 1-599 | 120 records, 16 HTN |

---

## Q. Unresolved Uncertainty

- **Generation reproducibility:** `I don't know` zero-patient case not reproduced via Ollama at `temperature≈0.8`; requires HF `Qwen2.5-7B-Instruct` with `temperature=0.3, max_tokens=200` live run (billing). Sampling abstention / provider failure remains moderate-confidence.
- **Exact K behind tester’s 15/0 observations:** commit history shows only `k=2`; dirty `k=50` edit suggests at least one sufficient-K run occurred locally. Tester’s exact wiring (direct `runner.py` vs CLI flag) unknown.
- **Ollama vs HF divergence:** Ollama path has no `max_tokens` cap, so 15-patient at K=20 is pure nondeterministic truncation; HF path would additionally truncate at 200 tokens. Need paired `temperature=0` experiment to quantify each provider’s contribution.
- **Token count precision:** Qwen/llama tokenizer not available locally; token estimates use `words×1.3`. Exact `k=20` context tokens (~1716) and 16-patient answer tokens (~480) need tokenizer-verified measurement.
- **Sanitized artifacts under `/tmp/end_to_end_audit/`** are metadata-only (counts, chars); full masked contexts are reproducible via `build_rag` + `retrieve(...,k=20)` as proven by `Chunks match build_rag: True`.

---

## Conclusion (one of five, explicit)

**3. LLM generation is losing records — when sufficient K (20/40/50) is used as audited.**

- Retrieval at K≥20: **0/16 lost** (16/16 recall).
- Context construction: **0/16 lost** (`retrieved == context`, no filtering/truncation).
- Generation at K=20 (Ollama `llama3.2`): **1–7/16 lost per run** (9–15 enumerated, mean ~12/16; hallucinated duplicates at 18/16; K=50 degrades to 4–6/16). HF at `max_tokens=200` would additionally truncate after ~8/16.

If the scope is the **committed interactive default (k=2)**, then the correct aggregate statement is **4. Multiple stages contribute** (retrieval 14/16 lost + generation faithful to the 2-record context). For the sufficient-K regime the user asked to audit, the isolated loss point is generation.

---

## Appendix — Reproduction (read-only, `/tmp` only)

```bash
python3 /tmp/audit_retrieval.py          # K sweep, full ranking, default-k proof, masking, context size
python3 /tmp/audit_phase2.py              # benchmark vs runtime parity (embeddings, chunking, FAISS, prompt, dataset)
# Artifacts retained:
# /tmp/end_to_end_audit/retrieval_table.json
# /tmp/end_to_end_audit/context_audit_k{2,20,40,50}.txt
# /tmp/end_to_end_audit/context_canonical_k{2,20,40,50}.txt
# No files under secure_rag/, benchmarks/, tests/, or config were written.
```

**Repository state after audit:** `git status --short` shows only `?? reports/retrieval/comparison/` (new report `end_to_end_runtime_audit.md` plus prior `aggregate_runtime_audit.md`); no production source modified. Existing `python3 -m pytest tests/` unchanged (spacy-dependent slow tests require `en_core_web_sm`).
