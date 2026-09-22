# Secure-RAG Experimental Summary

## 1. Research Objective
Evaluate privacy-aware RAG with pre-embedding masking: does masking preserve retrieval utility, and where does aggregate pipeline lose coverage (retrieval depth vs generation)? All experiments are characterization, not yet adaptive retrieval.

## 2. Dataset
- **120 records** deterministic synthetic Indian hospital (seed 42), 33 diseases, 107 treatments, 1:1 chunk/record (Words 51–83 <500 → 120 chunks).
- **605 queries** in `ground_truth_v2.json`: 601 single-target (1 relevant) + 4 true multi-record (7–20) +1 single-target aggregate (T2D_HYP 1). Ground truth validated; regenerable via `data/generate_dataset.py` + `ground_truth.py`.
- **Artifacts:** `reports/dataset/validation_report.md`, `distribution_statistics.md`, `data/dataset_architecture.md` → paper `02_dataset.md`.

## 3. Privacy / Pre-Embedding Masking
- **Mechanism:** `medical` policy + `medical` detector stack (`RegexDetector` + `DomainDetector` + `SpaCyDetector`) via `mask_text` per record **before** chunking/embedding (`secure_rag/rag_pipeline.py:66-77`). Raw sensitive never enters FAISS.
- **Production defaults unchanged:** `retriever.py` k=2, `embedding` `all-MiniLM-L6-v2`, `vector_store` FAISS `IndexFlatL2`, `rag_pipeline` flow, `masker` medical masking.
- **Query never masked.**
- **Result:** Per-query limited changes while aggregate preserved. Dense at k=10: 25 degraded /26 improved /554 unchanged. BM25 raw vs masked identical. See `reports/comparison/summary.md` §9.

## 4. Dense Retrieval
- **Baseline:** FAISS masked, K=1–50.
- **Overall:** k=10 HitRate0.0909 Prec0.0145 Rec0.0888 MRR0.0325; k=50 Rec0.4215.
- **Single (601):** poor 0.0849 at k10 → generic query design, not algorithm failure.
- **Multi aggregates:** scales with k; 0.325 at k2 →0.73 at k10 →0.98 at k20.
- **Paper:** `03_dense_retrieval.md`, `reports/retrieval/dense/depth_report.md`.

## 5. BM25 Comparison
- **Method:** BM25Okapi (rank-bm25) lexical, same masking/ground-truth. Raw vs masked both indices.
- **Finding:** Competitive, **no meaningful overall improvement** over Dense. At k10: Rec0.0887 MRR0.0308 vs Dense 0.0888/0.0325; k50 both 0.4215. Raw vs masked identical. See `04_bm25.md` (worktree experiment/bm25).

## 6. Hybrid Comparison
- **Method:** Dense+BM25 RRF (RRF_K=60). Masked only.
- **Finding:** **Dense/Hybrid tie** (k10 0.0888/0.0325 identical), adds complexity for narrow benefit. No raw Hybrid to verify masking claim (21-degradation claim unsupported). See `05_hybrid.md` (worktree experiment/hybrid).

## 7. Retrieval Depth
- **K evaluated:** 2,5,10,20,30,50.
- **Per-aggregate recall:** HYP 0.125→1.0 at k20; AMLO 0.118→0.941 at k20→1.0 at k30; PARA 0.10→0.95 at k20→1.0 at k30; MET 0.286→1.0 at k10.
- **Aggregates:** Mean multi-record recall 0.157(k2)→0.393(k5)→0.678(k10)→0.973(k20)≈**97.8%**→1.0(k30) **=100%**. Precision decays 1.0→0.244.
- **Context:** linear growth 1k→27k chars, prompt tokens 366→4095 cap.
- **Paper:** `06_retrieval_depth.md`.

## 8. End-to-End Runtime Audit
- **Audits:** `reports/retrieval/comparison/aggregate_runtime_audit.md` + `end_to_end_runtime_audit.md` — measured same retrievals as §7, confirmed production K=2 →2/16 recall 0.125, K=20→16/16, K=40/50→16/16, context construction preserves all retrieved (hash verified), generation loss beyond retrieval (see §9–11).

## 9. Generation Budget (v2, Valid)
- **Frozen:** K=20 16/16 hypertension, 20 masked chunks (16+4 distractors), Ollama llama3.2 temp0.3, fingerprint measurand (threshold ≥12), provider `done_reason`/`eval_count` via `secure_rag/generator.py` instrumentation.
- **Budgets ×5:** 200→**0.175**±0.025 missing13.2 unsupported0.8 trunc1.0 | 400→**0.412** | 600→**0.562** | 800→**0.737** trunc0.40 | 1200→**0.825** trunc0.60. Max 14/16 (0.875), no 16/16. All 200–600 `length`; 800–1200 mix `stop` (705–1113 tokens) still miss 2–3.
- **Verdict:** Significant bottleneck ≤600, partially at 800–1200 where recall saturates <1.0; remaining loss after retrieval. Unsupported 0.8→3.6 with budget.
- **Historical:** v1 invalid (0.0 due to masked MRN regex); preserved but not paper.
- **Paper:** `07_generation_budget.md`, artifacts `generation_results_v2.json`/`generation_metrics_v2.json`.

## 10. Generation Filtering
- **Intervention:** Only records where Diagnosis contains Hypertension instruction (+51 prompt tokens, context hash identical 93eb5f…).
- **10 runs:** 800→0.425 vs baseline 0.738 **Δ−0.312**, unsupported 3.2→0.2 **Δ−3.0**, trunc 0.40→1.00; 1200→0.600 vs 0.825 **Δ−0.225**, unsupported 3.6→1.6 **Δ−2.0**.
- **Verdict:** Closest to B (precision up) but more negative — improves selection (fewer distractors) but **does not improve recall**, reduces patients-per-token (verbose bullets), increases truncation at 800. Not sufficient.
- **Paper:** `08_generation_filtering.md`.

## 11. Adaptive-K Characterization
- **6 K (2,5,10,20,30,50) ×5 queries (16,17,20,7,1) =30 runs** at 1200 budget, baseline prompt.
- **Retrieval:** mean rec 0.326→0.514→0.743→**0.978**→**1.0**→1.0; multi-rec 0.157→0.393→0.678→0.973→1.0→1.0 — confirms §7 (K=20≈97.8%, K=30=100%).
- **Generation:** At K where retrieval=1.0 but gen=0 (HYP k20, MET k10+), loss is after retrieval. Only AMLO k20 shows 14/17 (0.824) peak; beyond 20 distractors degrade generation (k30/50→0). All 30 runs `stop` (no truncation at 1200 here due to short answers 13–573 tokens) — generation frequently remains bottleneck.
- **Paper:** `09_adaptive_k_characterization.md`.

## 12. Cross-Experiment Findings
- Pre-embedding masking **approximately preserves** aggregate retrieval; Dense remains best tradeoff (BM25/Hybrid not meaningfully better).
- **K=2 insufficient** for multi-record (0.157 multi recall); K=20 gives near-complete, K=30 gives 100% but precision collapses.
- **Generation budget** is major bottleneck ≤600, but even at 1200 retrieval-complete pipeline does not reach 16/16; budget alone insufficient.
- **Filtering** reduces unsupported but not recall — verbosity reduces efficiency.
- **K characterization** shows large K adds distractors without reliably helping generation; generation is frequent remaining bottleneck.

## 13. Limitations
- Single synthetic dataset (120), 5 aggregates; single-record poor due to templated queries.
- Single provider/model (Ollama llama3.2) temp0.3; 5 reps (budget/filter) or 1 rep (adaptive-K) — nondeterminism limited characterization.
- Fingerprint proxy (age+diagnosis) not clinician adjudication; may over/undercount.
- BM25 params not tuned; Hybrid no raw baseline; tokenizer simple.
- No adaptive retrieval implemented — characterization only.

## 14. Final Defensible Claims
- **Privacy:** Pre-embedding medical masking prevents raw PII entering vector store while **approximately preserving** retrieval utility on evaluated dataset.
- **Retrieval:** Dense competitive baseline; BM25/Hybrid do not substantially outperform Dense; masking limited per-query changes.
- **Depth:** Multi-record needs K≈20 for ~98% mean recall, 30 for 100%; K=2 insufficient; precision/context tradeoff linear.
- **Generation:** Budget is significant bottleneck (0.175→0.825 from 200→1200) but saturates below exhaustive; remaining 2–3 missing at 1200 not solely budget; generation frequently remains bottleneck even when retrieval=1.0.
- **Filtering:** Instruction reduces unsupported enumeration but does not improve recall at tested budgets; may reduce token efficiency.
- **Systems:** No adaptive-K implemented; K passed explicitly from research runner only; production default K=2 preserved.

Do **not** claim: filtering improves recall, Hybrid improves retrieval, BM25 superior, adaptive retrieval implemented, K=20 is implemented adaptive policy, or larger budget alone achieves 16/16.

## 15. Results Still Needed
- **Only pending:** Adaptive retrieval implementation + evaluation (compare adaptive vs fixed-K on recall/precision/latency); generalization to other aggregate queries; larger budgets >1200 or concise enumeration prompt; other temperatures/models.

---

## Experiment Status Table

| Experiment | Status | Main finding | Paper relevance |
|------------|--------|--------------|-----------------|
| Dataset | Complete | 120 records /605 queries deterministic | Foundation |
| Dense retrieval | Complete | Strong baseline, masking preserves aggregate | Baseline |
| BM25 | Complete | Competitive, no meaningful gain (k10 0.0887 vs 0.0888) | Comparator — keep |
| Hybrid | Complete | Tie with Dense (k10 0.0888), adds complexity | Comparator — keep |
| Retrieval depth | Complete | K=20≈97.8% multi, K=30=100%, precision tradeoff | Design insight |
| End-to-end runtime audit | Complete | Production K2=2/16; K20=16/16; loss after retrieval | Diagnostics |
| Generation budget v2 | Complete | 0.175→0.825 saturates <1.0; budget partially bottleneck | Bottleneck evidence |
| Generation filtering | Complete | Recall −0.31 at 800, unsupported −3.0 (precision up recall down) | Negative result — keep |
| Adaptive-K characterization | Complete | Retrieval K=20≈97.3% multi but generation often fails even when retrieval=1.0 | Motivation for adaptive retrieval |

**No recalculated results; numbers preserved verbatim from original reports. Inconsistencies flagged in §14 notes (e.g., Hybrid 21-degradations unsupported).**
