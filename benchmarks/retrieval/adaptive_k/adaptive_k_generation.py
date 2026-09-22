#!/usr/bin/env python3
"""
Adaptive Retrieval Depth — Generation (research-only)

Performs end-to-end generation for 5 aggregate queries x 6 K values = 30 runs.

Constraints:
- Provider: Ollama, Model: llama3.2:latest, Temperature: 0.3, Budget: 1200 tokens
- Prompt: baseline v2 (no filtering instruction)
- Build masked index once, retrieve per K, assemble context, call LLM
- Verify: retrieved records == records included in context, raw MRNs absent
- Metrics: reuse corrected generation v2 masked-record fingerprint evaluator
- Truncation via done_reason == "length" → True, "stop" → False, missing → None
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from benchmarks.retrieval.ground_truth import AGGREGATE_QUERIES_V2
from secure_rag.detection import load_detector_stack
from secure_rag.embedding import embed_chunks
from secure_rag.generator import (
    LLM_PROVIDER,
    _build_ollama_prompt,
    generate_answer,
    get_last_ollama_metadata,
)
from secure_rag.masker import mask_text
from secure_rag.pdf_loader import chunk_text
from secure_rag.policies import load_policy
from secure_rag.vector_store import VectorStore

# Reuse v2 masked evaluator
from benchmarks.retrieval.generation.generation_metrics import compute_masked_generation_metrics

RETRIEVAL_DIR = Path(__file__).parent
PROJECT_DIR = Path(__file__).parent.parent.parent.parent
RESULTS_PATH = RETRIEVAL_DIR / "adaptive_k_generation_results.json"

K_VALUES = [2, 5, 10, 20, 30, 50]
AGG_QIDS = ["AGG_HYPERTENSION", "AGG_AMLODIPINE_5MG", "AGG_PARACETAMOL_650MG", "AGG_METFORMIN_500MG", "AGG_T2D_HYPERTENSION"]

# Generation constraints
PROVIDER = "ollama"
MODEL = "llama3.2:latest"
TEMPERATURE = 0.3
GENERATION_BUDGET = 1200

# Baseline v2 prompt — do not add filtering instruction


def _load_mrn_records_raw() -> Dict[str, dict]:
    text = Path(PROJECT_DIR / "data" / "sample_patient_data.txt").read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]
    records: Dict[str, dict] = {}
    for block in blocks:
        m = re.search(r"Medical ID:\s*(MRN\d+)", block)
        if not m:
            continue
        rid = m.group(1)
        records[rid] = {"record_id": rid, "text": block}
    return records


def _build_masked_index(records: Dict[str, dict]):
    policy = load_policy("medical")
    detectors = load_detector_stack("medical")
    texts_with_ids = []
    for rid in sorted(records.keys()):
        r = records[rid]
        text = mask_text(r["text"], policy=policy, detectors=detectors)
        texts_with_ids.append((rid, text))
    chunks: List[str] = []
    chunk_record_map: List[str] = []
    for rid, text in texts_with_ids:
        record_chunks = chunk_text(text)
        chunks.extend(record_chunks)
        chunk_record_map.extend([rid] * len(record_chunks))
    embeddings = embed_chunks(chunks)
    vector_store = VectorStore(embeddings)
    return vector_store, chunks, chunk_record_map


def _retrieve_top_k(query: str, vector_store: VectorStore, k: int):
    q_vec = embed_chunks([query])
    q_vec = np.array(q_vec).astype("float32")
    distances, indices = vector_store.search(q_vec, k=k)
    if isinstance(indices, list) and len(indices) > 0 and isinstance(indices[0], list):
        indices = indices[0]
    if hasattr(distances, "tolist"):
        distances_list = distances[0].tolist() if hasattr(distances, "shape") and getattr(distances, "ndim", 1) > 1 else distances.tolist()
        if isinstance(distances_list, list) and len(distances_list) > 0 and isinstance(distances_list[0], list):
            distances_list = distances_list[0]
    else:
        distances_list = list(distances)
    indices_list = indices if isinstance(indices, list) else list(indices)
    distances_list = distances_list if isinstance(distances_list, list) else list(distances_list)
    return indices_list, distances_list


def _assemble_context(chunks: List[str], retrieved: List[Dict]) -> tuple[str, List[str], List[str]]:
    retrieved_chunks: List[str] = []
    retrieved_ids: List[str] = []
    for item in retrieved:
        idx = item["chunk_index"]
        if 0 <= idx < len(chunks):
            retrieved_chunks.append(chunks[idx])
            retrieved_ids.append(item["record_id"])
    context = "\n\n".join(ch for ch in retrieved_chunks if ch)
    return context, retrieved_chunks, retrieved_ids


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=False, local_files_only=True)
        return len(tok.encode(text))
    except Exception:
        pass
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        pass
    words = len(text.split())
    return int(words * 1.3) + 1


def _verify_no_raw_mrns(text: str) -> Dict:
    pattern = re.compile(r"\bMRN\d+\b")
    found = pattern.findall(text)
    return {
        "raw_mrns": found,
        "raw_mrn_count": len(found),
        "contains_raw": len(found) > 0,
        "has_masked_placeholder": "[PATIENT_ID_MASKED]" in text,
    }


def _call_generation(context: str, query: str, budget: int = GENERATION_BUDGET, temperature: float = TEMPERATURE):
    """Call Ollama with controlled budget, capture provider metadata."""
    # Ensure LLM_PROVIDER is ollama
    # Use baseline v2 prompt (no filter)
    prompt_str = _build_ollama_prompt(context, query)
    system_prompt = None
    user_prompt = prompt_str
    messages = None
    model = os.getenv("OLLAMA_MODEL", MODEL)

    # The query passed to generate_answer is f"{query}\n\nAnswer:"
    start = time.time()
    answer = "".join(generate_answer(context, f"{query}\n\nAnswer:", max_tokens=budget, num_predict=budget, temperature=temperature))
    latency_ms = int((time.time() - start) * 1000)

    ollama_meta: Dict = {}
    try:
        ollama_meta = get_last_ollama_metadata() or {}
    except Exception:
        ollama_meta = {}

    # Token accounting: prefer provider counts
    output_tokens = None
    output_tokens_is_estimated = True
    prompt_tokens = None
    prompt_tokens_is_estimated = True

    if "eval_count" in ollama_meta and ollama_meta["eval_count"] is not None:
        try:
            output_tokens = int(ollama_meta["eval_count"])
            output_tokens_is_estimated = False
        except Exception:
            output_tokens = _estimate_tokens(answer)
            output_tokens_is_estimated = True
    else:
        output_tokens = _estimate_tokens(answer)
        output_tokens_is_estimated = True

    if "prompt_eval_count" in ollama_meta and ollama_meta["prompt_eval_count"] is not None:
        try:
            prompt_tokens = int(ollama_meta["prompt_eval_count"])
            prompt_tokens_is_estimated = False
        except Exception:
            prompt_tokens = None
            prompt_tokens_is_estimated = True
    else:
        if prompt_str:
            prompt_tokens = _estimate_tokens(prompt_str)
            prompt_tokens_is_estimated = True
        else:
            prompt_tokens = None
            prompt_tokens_is_estimated = True

    # Truncation via done_reason
    finish_reason = None
    truncation = None
    truncation_status = "unknown"
    done_reason = ollama_meta.get("done_reason") if ollama_meta else None

    if done_reason is not None:
        if done_reason == "length":
            finish_reason = "length"
            truncation = True
            truncation_status = "truncated"
        elif done_reason == "stop":
            finish_reason = "stop"
            truncation = False
            truncation_status = "not_truncated"
        else:
            finish_reason = str(done_reason)
            truncation = False
            truncation_status = "not_truncated"
    else:
        if ollama_meta.get("done") is True:
            finish_reason = "unknown"
            truncation = None
            truncation_status = "unknown"
        else:
            finish_reason = "unknown"
            truncation = None
            truncation_status = "unknown"

    # Context sizes
    context_char_count = len(context)
    context_word_count = len(context.split())

    return {
        "answer_text": answer,
        "finish_reason": finish_reason,
        "done_reason": done_reason,
        "truncation": truncation,
        "truncation_status": truncation_status,
        "output_tokens": output_tokens,
        "output_tokens_is_estimated": output_tokens_is_estimated,
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_is_estimated": prompt_tokens_is_estimated,
        "latency_ms": latency_ms,
        "provider": LLM_PROVIDER,
        "model": model,
        "temperature": temperature,
        "prompt": prompt_str,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "messages": messages,
        "ollama_metadata": ollama_meta,
        "context_char_count": context_char_count,
        "context_word_count": context_word_count,
    }


def run_generation() -> dict:
    print("=" * 60)
    print("Adaptive K — Generation Experiment")
    print("=" * 60)
    print(f"K values: {K_VALUES}")
    print(f"Queries: {AGG_QIDS}")
    print(f"Provider: {PROVIDER}, Model: {MODEL}, Temp: {TEMPERATURE}, Budget: {GENERATION_BUDGET}")

    # Force Ollama provider check
    assert LLM_PROVIDER == "ollama", f"LLM_PROVIDER must be ollama, got {LLM_PROVIDER}"

    mrn_records = _load_mrn_records_raw()
    print(f"\nLoaded {len(mrn_records)} MRN records")
    assert len(mrn_records) == 120

    print("\nBuilding masked index...")
    vector_store, chunks, chunk_record_map = _build_masked_index(mrn_records)
    print(f"Index: {len(chunks)} chunks")

    gt_map = {q["qid"]: q for q in AGGREGATE_QUERIES_V2}

    runs: List[Dict] = []
    for qid in AGG_QIDS:
        gt_entry = gt_map[qid]
        question = gt_entry["question"]
        relevant_records = sorted(gt_entry["relevant_records"])
        gt_set = set(relevant_records)
        print(f"\n=== {qid}: {question} ({len(relevant_records)} relevant) ===")

        for k in K_VALUES:
            print(f"\n  [K={k}] Retrieving...")
            indices_list, distances_list = _retrieve_top_k(question, vector_store, k=k)
            retrieved = []
            for rank, (chunk_idx, score) in enumerate(zip(indices_list, distances_list)):
                rid = chunk_record_map[chunk_idx] if 0 <= chunk_idx < len(chunk_record_map) else "UNKNOWN"
                retrieved.append({
                    "chunk_index": int(chunk_idx),
                    "score": round(float(score), 4),
                    "record_id": rid,
                    "rank": rank,
                    "relevant": rid in gt_set,
                })
            # Assemble context
            context, context_chunks, retrieved_ids = _assemble_context(chunks, retrieved)
            # Integrity: retrieved records == records included in context
            assert retrieved_ids == [r["record_id"] for r in retrieved], "Context assembly mismatch"
            assert len(retrieved_ids) == k, f"Context record count {len(retrieved_ids)} != K {k}"
            assert len(context_chunks) == k

            # Privacy checks
            ctx_audit = _verify_no_raw_mrns(context)
            # raw MRNs must be 0
            if ctx_audit["contains_raw"]:
                raise RuntimeError(f"Raw MRNs leaked into context for {qid} K={k}: {ctx_audit['raw_mrns']}")
            # Also verify prompt has no raw MRNs (will check after generation)

            # Generation
            print(f"  Generating with budget={GENERATION_BUDGET}...")
            gen = _call_generation(context, question, budget=GENERATION_BUDGET, temperature=TEMPERATURE)

            # Verify prompt privacy
            prompt_audit = _verify_no_raw_mrns(gen["prompt"])
            if prompt_audit["contains_raw"]:
                raise RuntimeError(f"Raw MRNs leaked into prompt for {qid} K={k}: {prompt_audit['raw_mrns']}")

            # Compute generation metrics via masked fingerprint evaluator
            gm = compute_masked_generation_metrics(gen["answer_text"], context_chunks, retrieved_ids, relevant_records, k=k)

            # Record retrieval metrics for this K as well for convenience
            # Compute retrieval recall for table
            seen_relevant = set(rid for rid in retrieved_ids if rid in gt_set)
            retrieval_recall = len(seen_relevant) / len(gt_set) if gt_set else 0.0
            retrieval_precision = len(seen_relevant) / k if k else 0.0
            hit_rate = 1 if seen_relevant else 0
            # MRR: first relevant rank
            mrr = 0.0
            for idx, rid in enumerate(retrieved_ids):
                if rid in gt_set:
                    mrr = 1.0 / (idx + 1)
                    break

            run_entry = {
                "qid": qid,
                "question": question,
                "k": k,
                "generation_budget": GENERATION_BUDGET,
                "retrieved_chunk_indices": [int(x) for x in indices_list],
                "retrieved_record_ids": retrieved_ids,
                "retrieved_details": retrieved,
                "relevant_record_ids": relevant_records,
                "scores": [round(float(s), 4) for s in distances_list],
                "retrieval_metrics": {
                    "hit_rate": hit_rate,
                    "precision": retrieval_precision,
                    "recall": retrieval_recall,
                    "mrr": mrr,
                    "relevant_retrieved": len(seen_relevant),
                    "total_relevant": len(relevant_records),
                },
                "context": context,
                "context_chunks": context_chunks,
                "context_char_count": gen["context_char_count"],
                "context_word_count": gen["context_word_count"],
                "prompt": gen["prompt"],
                "prompt_char_count": len(gen["prompt"]),
                "prompt_tokens": gen["prompt_tokens"],
                "prompt_tokens_is_estimated": gen["prompt_tokens_is_estimated"],
                "retrieved_record_count": len(retrieved_ids),
                "context_integrity": {
                    "retrieved_equals_context": retrieved_ids == [r["record_id"] for r in retrieved],
                    "context_record_count": len(context_chunks),
                    "k": k,
                    "match": len(context_chunks) == k and len(retrieved_ids) == k,
                },
                "privacy": {
                    "raw_mrns_in_context": ctx_audit["raw_mrns"],
                    "raw_mrn_count_context": ctx_audit["raw_mrn_count"],
                    "raw_mrns_in_prompt": prompt_audit["raw_mrns"],
                    "raw_mrn_count_prompt": prompt_audit["raw_mrn_count"],
                    "has_masked_placeholder_context": ctx_audit["has_masked_placeholder"],
                    "has_masked_placeholder_prompt": prompt_audit["has_masked_placeholder"],
                },
                "provider": gen["provider"],
                "model": gen["model"],
                "temperature": gen["temperature"],
                "answer_text": gen["answer_text"],
                "answer_char_count": len(gen["answer_text"]),
                "answer_word_count": len(gen["answer_text"].split()),
                "finish_reason": gen["finish_reason"],
                "done_reason": gen["done_reason"],
                "truncation": gen["truncation"],
                "truncation_status": gen["truncation_status"],
                "output_tokens": gen["output_tokens"],
                "output_tokens_is_estimated": gen["output_tokens_is_estimated"],
                "latency_ms": gen["latency_ms"],
                "ollama_metadata": gen["ollama_metadata"],
                # Generation metrics
                "generation_metrics": gm,
                "relevant_records_found": gm["relevant_records_found"],
                "generation_recall": gm["generation_recall"],
                "missing_records": gm["missing_records"],
                "missing_count": gm["missing_count"],
                "duplicate_records": gm["duplicate_records"],
                "duplicate_mrns": gm["duplicate_mrns"],
                "unsupported_records": gm["unsupported_records"],
                "unsupported_count": gm["unsupported_count"],
                "found_mrns": gm["found_mrns"],
                "found_mrns_ordered": gm["found_mrns_ordered"],
                "found_gt_ordered": gm["found_gt_ordered"],
                "enumerated_patient_count": gm["enumerated_patient_count"],
                "hallucinated_count": gm["hallucinated_count"],
                "segment_count": gm["segment_count"],
            }
            print(f"    -> retrieval recall {retrieval_recall:.3f} ({len(seen_relevant)}/{len(relevant_records)}), gen recall {gm['generation_recall']:.3f} ({gm['relevant_records_found']}/{len(relevant_records)}), enum {gm['enumerated_patient_count']}, unsupp {gm['unsupported_count']}, trunc {gen['truncation_status']} tokens {gen['output_tokens']}")
            runs.append(run_entry)

    output = {
        "version": "adaptive_k_v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "description": "Adaptive K generation experiment: 5 queries x 6 K values =30 runs, Ollama llama3.2 1200 tokens, baseline v2 prompt, masked context, privacy verified, masked fingerprint evaluator.",
        "k_values": K_VALUES,
        "queries": AGG_QIDS,
        "generation_budget": GENERATION_BUDGET,
        "provider": PROVIDER,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "total_runs": len(runs),
        "runs": runs,
        "privacy_verified": all(r["privacy"]["raw_mrn_count_context"] == 0 and r["privacy"]["raw_mrn_count_prompt"] == 0 for r in runs),
        "context_integrity_verified": all(r["context_integrity"]["match"] for r in runs),
    }
    return output


def save_results(results: dict, path=None) -> Path:
    if path is None:
        path = RESULTS_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def load_results(path=None) -> dict:
    if path is None:
        path = RESULTS_PATH
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Adaptive K generation runner")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    res = run_generation()
    out = save_results(res, path=args.output)
    print(f"\nResults saved to {out}")
    print(f"Total runs: {res['total_runs']}")
    print(f"Privacy verified: {res['privacy_verified']}")
    print(f"Context integrity: {res['context_integrity_verified']}")
