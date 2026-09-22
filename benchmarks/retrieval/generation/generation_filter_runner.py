#!/usr/bin/env python3
"""
Controlled generation-filtering experiment — research-only.

Tests hypothesis: explicitly instructing LLM to enumerate ONLY records whose
Diagnosis contains Hypertension improves generation-stage recall and reduces
unsupported enumeration.

Design:
- Canonical query: "Give me all patients with hypertension." (16 GT)
- Dense FAISS, medical pre-embedding masking, K=20, retrieval ONCE, freeze context.
- Reuse exact production context construction (must match v2 hash).
- ONLY change: generation instruction added.
- Vary budgets: [800,1200] x 5 reps = 10 generations.
- Fixed: query, retrieved records, context ordering, model, provider, temperature 0.3.
- Instrumentation: Ollama final metadata, provider eval_count, done_reason.
"""

import hashlib
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
from secure_rag.generator import (
    LLM_PROVIDER,
    _build_hf_messages,
    _build_ollama_prompt,
    generate_answer,
    get_last_ollama_metadata,
)
from secure_rag.masker import mask_text
from secure_rag.pdf_loader import chunk_text
from secure_rag.policies import load_policy
from secure_rag.vector_store import VectorStore
from secure_rag.embedding import embed_chunks

from .generation_metrics import (
    EXPECTED_HYPERTENSION_COUNT,
    EXPECTED_K,
    compute_masked_generation_metrics,
)

RETRIEVAL_DIR = Path(__file__).parent.parent
GENERATION_DIR = Path(__file__).parent
PROJECT_DIR = GENERATION_DIR.parent.parent.parent

CANONICAL_QUERY = "Give me all patients with hypertension."
GENERATION_BUDGETS = [800, 1200]
REPS_PER_BUDGET = 5
K = 20
TEMPERATURE = 0.3

# Filtering instruction – ONLY change vs v2
FILTER_INSTRUCTION = (
    "Enumerate only records whose Diagnosis contains Hypertension. "
    "Do not enumerate retrieved records whose Diagnosis does not contain Hypertension. "
    "Do not infer, combine, or invent patient records. "
    "Each enumerated patient must correspond to a retrieved record whose Diagnosis explicitly contains Hypertension."
)

# Ground truth
_HYP_GT_ENTRY = next(q for q in AGGREGATE_QUERIES_V2 if q["qid"] == "AGG_HYPERTENSION")
GROUND_TRUTH_MRNS: List[str] = sorted(_HYP_GT_ENTRY["relevant_records"])

RESULTS_PATH = GENERATION_DIR / "generation_filter_results.json"
METRICS_PATH = GENERATION_DIR / "generation_filter_metrics.json"
# Reference v2 results for context identity check
V2_RESULTS_PATH = GENERATION_DIR / "generation_results_v2.json"


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
    """Build index using medical pre-embedding masking path (reuse production masking)."""
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


def _retrieve_once(vector_store: VectorStore, chunk_record_map: List[str], query: str, k: int = 20):
    q_vec = embed_chunks([query])
    q_vec = np.array(q_vec).astype("float32")
    distances, indices = vector_store.search(q_vec, k=k)
    indices_list = indices if isinstance(indices, list) else indices[0].tolist() if hasattr(indices, "tolist") else list(indices)
    distances_list = distances[0].tolist() if hasattr(distances, "shape") and distances.ndim > 1 else distances.tolist() if hasattr(distances, "tolist") else list(distances)
    if isinstance(distances_list, list) and len(distances_list) > 0 and isinstance(distances_list[0], list):
        distances_list = distances_list[0]
    if isinstance(indices_list, list) and len(indices_list) > 0 and isinstance(indices_list[0], list):
        indices_list = indices_list[0]
    retrieved = []
    for rank, (chunk_idx, score) in enumerate(zip(indices_list, distances_list)):
        rid = chunk_record_map[chunk_idx] if 0 <= chunk_idx < len(chunk_record_map) else "UNKNOWN"
        retrieved.append(
            {
                "chunk_index": int(chunk_idx),
                "score": round(float(score), 4),
                "record_id": rid,
                "rank": rank,
            }
        )
    return retrieved, distances_list


def _assemble_context(chunks: List[str], retrieved: List[Dict]) -> tuple[str, List[str], List[str]]:
    """Assemble context string and parallel IDs in retrieval order."""
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


def _build_filtered_ollama_prompt(context: str, query: str) -> str:
    """
    Build filtered prompt: baseline prompt + filtering instruction.

    Baseline: _build_ollama_prompt(context, query) = 
        "You are a RAG assistant... Context:\n{context}\n\nQuestion:\n{query}"

    Filtered: same context, but query augmented with FILTER_INSTRUCTION.
    We inject instruction as separate paragraph after the question so context remains identical.
    The query arg is expected to be CANONICAL_QUERY (without Answer suffix);
    we construct "Question: {query}\n\n{filter}\n\nAnswer:".
    """
    # If query already contains FILTER_INSTRUCTION or Answer, strip to canonical
    # But for robustness, if FILTER_INSTRUCTION already in query, don't double-add.
    if FILTER_INSTRUCTION in query:
        filtered_query = query
    else:
        # Remove trailing "Answer:" if present to normalize, then add filter + Answer
        q = query.strip()
        # Strip trailing Answer markers for clean construction
        if q.endswith("Answer:"):
            q = q[: -len("Answer:")].rstrip()
            # Also strip extra Answer duplicates
            while q.endswith("Answer:"):
                q = q[: -len("Answer:")].rstrip()
        # q should now be canonical query
        filtered_query = f"{q}\n\n{FILTER_INSTRUCTION}\n\nAnswer:"
        # If original query already had Answer suffix, we replaced it; if not, we added it.
    return _build_ollama_prompt(context, filtered_query)


def _build_filtered_hf_messages(context: str, query: str):
    if FILTER_INSTRUCTION in query:
        filtered_query = query
    else:
        q = query.strip()
        if q.endswith("Answer:"):
            q = q[: -len("Answer:")].rstrip()
            while q.endswith("Answer:"):
                q = q[: -len("Answer:")].rstrip()
        filtered_query = f"{q}\n\n{FILTER_INSTRUCTION}\n\nAnswer:"
    return _build_hf_messages(context, filtered_query)


def _call_generation_filtered(context: str, query: str, budget: int, temperature: float = TEMPERATURE):
    """Call production generator with filtered instruction and capture provider metadata.

    Context is unchanged from v2. Only prompt (query augmented) differs.
    Query arg should be CANONICAL_QUERY (or with Answer suffix – normalized internally).
    """
    provider = LLM_PROVIDER
    model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct") if provider != "ollama" else os.getenv("OLLAMA_MODEL", "llama3.2:latest")

    # Normalize query to canonical + filter + Answer for consistent prompt
    q_norm = query.strip()
    if FILTER_INSTRUCTION in q_norm:
        filtered_query_for_gen = q_norm if q_norm.endswith("Answer:") else f"{q_norm}\n\nAnswer:"
    else:
        # strip trailing Answer markers
        q_base = q_norm
        while q_base.endswith("Answer:"):
            q_base = q_base[: -len("Answer:")].strip()
            # also strip possible double newline left
        # Re-add filter and Answer exactly once
        filtered_query_for_gen = f"{q_base}\n\n{FILTER_INSTRUCTION}\n\nAnswer:"

    if provider == "ollama":
        # Filtered prompt uses normalized filtered query
        prompt_str = _build_ollama_prompt(context, filtered_query_for_gen)
        system_prompt = None
        user_prompt = prompt_str
        messages = None
    else:
        messages = _build_hf_messages(context, filtered_query_for_gen)
        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]
        prompt_str = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"

    start = time.time()
    # Pass filtered query to generator so LLM sees instruction
    answer = "".join(generate_answer(context, filtered_query_for_gen, max_tokens=budget, num_predict=budget, temperature=temperature))
    latency_ms = int((time.time() - start) * 1000)

    # Capture provider metadata (Ollama)
    ollama_meta: Dict = {}
    if provider == "ollama":
        try:
            ollama_meta = get_last_ollama_metadata() or {}
        except Exception:
            ollama_meta = {}

    # Token accounting: prefer provider counts
    output_tokens = None
    output_tokens_is_estimated = True
    prompt_tokens = None
    prompt_tokens_is_estimated = True

    if provider == "ollama" and "eval_count" in ollama_meta and ollama_meta["eval_count"] is not None:
        try:
            output_tokens = int(ollama_meta["eval_count"])
            output_tokens_is_estimated = False
        except Exception:
            output_tokens = _estimate_tokens(answer)
            output_tokens_is_estimated = True
    else:
        output_tokens = _estimate_tokens(answer)
        output_tokens_is_estimated = True if provider == "ollama" else True

    if provider == "ollama" and "prompt_eval_count" in ollama_meta and ollama_meta["prompt_eval_count"] is not None:
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

    # Truncation / finish_reason based on provider metadata
    finish_reason = None
    truncation = None
    truncation_status = "unknown"
    done_reason = ollama_meta.get("done_reason") if ollama_meta else None

    if provider == "ollama":
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
    else:
        finish_reason = "unknown"
        truncation = None
        truncation_status = "unknown"

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
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "prompt": prompt_str,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "messages": messages,
        "ollama_metadata": ollama_meta,
        "filter_instruction": FILTER_INSTRUCTION,
    }


def _verify_frozen_context(context: str, context_chunks: List[str]):
    """Audit that LLM context does not contain raw MRNs."""
    raw_mrn_pattern = re.compile(r"\bMRN\d+\b")
    found = raw_mrn_pattern.findall(context)
    has_masked = "[PATIENT_ID_MASKED]" in context
    return {
        "raw_mrns_in_context": found,
        "raw_mrn_count": len(found),
        "has_masked_placeholder": has_masked,
        "context_contains_raw": len(found) > 0,
    }


def _verify_prompt_no_raw_mrns(prompt: str) -> Dict:
    raw_mrn_pattern = re.compile(r"\bMRN\d+\b")
    found = raw_mrn_pattern.findall(prompt)
    has_filter = FILTER_INSTRUCTION in prompt
    return {
        "raw_mrns_in_prompt": found,
        "raw_mrn_count_prompt": len(found),
        "prompt_contains_raw": len(found) > 0,
        "has_filter_instruction": has_filter,
    }


def _context_hash(context: str) -> str:
    return hashlib.sha256(context.encode("utf-8")).hexdigest()


def run_pilot(budgets=(800, 1200)) -> Dict:
    """Mandatory 2-run pilot before full experiment (800 rep1, 1200 rep1).

    Verifies:
      1. Retrieval 20 / 16
      2. Context identical to v2
      3. No raw MRNs in context/prompt
      4. Filter instruction present
      5. Ollama metadata captured
      6. Evaluation succeeds
      7. No production files changed (implicit)
    """
    print("=" * 60)
    print("Generation-Filtering Pilot — 2 runs (800 & 1200 rep1)")
    print("=" * 60)
    mrn_records = _load_mrn_records_raw()
    vector_store, chunks, chunk_record_map = _build_masked_index(mrn_records)
    retrieved, _ = _retrieve_once(vector_store, chunk_record_map, CANONICAL_QUERY, k=K)
    retrieved_ids = [r["record_id"] for r in retrieved]
    seen = set()
    retrieved_ids_dedup = []
    for rid in retrieved_ids:
        if rid not in seen:
            seen.add(rid)
            retrieved_ids_dedup.append(rid)
    relevant_ids_dedup = [rid for rid in retrieved_ids_dedup if rid in GROUND_TRUTH_MRNS]
    assert len(retrieved) == EXPECTED_K, f"pilot: K must be 20, got {len(retrieved)}"
    assert len(relevant_ids_dedup) == EXPECTED_HYPERTENSION_COUNT, f"pilot: must have 16 relevant, got {len(relevant_ids_dedup)}"

    context, context_chunks, frozen_retrieved_ids = _assemble_context(chunks, retrieved)
    audit = _verify_frozen_context(context, context_chunks)
    print(f"Context audit: raw MRNs in context = {audit['raw_mrns_in_context']}, has [PATIENT_ID_MASKED]={audit['has_masked_placeholder']}")
    assert not audit["context_contains_raw"], f"pilot FAIL: raw MRNs found in LLM context: {audit['raw_mrns_in_context']}"
    assert audit["has_masked_placeholder"], "pilot FAIL: masked placeholder not found"

    # Context identity check vs v2
    ctx_hash = _context_hash(context)
    v2_hash = None
    if V2_RESULTS_PATH.exists():
        try:
            v2_data = json.loads(V2_RESULTS_PATH.read_text())
            v2_context = v2_data.get("retrieval", {}).get("context", "")
            if v2_context:
                v2_hash = _context_hash(v2_context)
                print(f"Context hash: filtered={ctx_hash[:16]}... v2={v2_hash[:16]}... match={ctx_hash==v2_hash}")
                assert ctx_hash == v2_hash, f"pilot FAIL: context hash mismatch filtered={ctx_hash} v2={v2_hash}"
            else:
                print("Warning: v2 context empty, skipping hash check")
        except Exception as e:
            print(f"Warning: v2 hash check failed: {e}")
    else:
        print("Warning: v2 results not found, skipping hash check")

    pilot_runs = []
    for budget in budgets:
        print(f"\n[Pilot] Generating budget={budget} rep=1 (filtered)...")
        gen = _call_generation_filtered(context, f"{CANONICAL_QUERY}\n\nAnswer:", budget, temperature=TEMPERATURE)
        answer_text = gen["answer_text"]
        assert answer_text and len(answer_text) > 20, f"pilot FAIL: answer empty for budget {budget}"
        # Verify prompt contains filter and no raw MRNs
        prompt_audit = _verify_prompt_no_raw_mrns(gen["prompt"])
        print(f"  prompt audit: has_filter={prompt_audit['has_filter_instruction']}, raw_in_prompt={prompt_audit['raw_mrns_in_prompt']}")
        assert prompt_audit["has_filter_instruction"], "pilot FAIL: filter instruction not in prompt"
        assert not prompt_audit["prompt_contains_raw"], f"pilot FAIL: raw MRNs in prompt: {prompt_audit['raw_mrns_in_prompt']}"
        # Also verify context still has no raw MRNs (double check)
        assert "[PATIENT_ID_MASKED]" in gen["prompt"], "pilot FAIL: masked placeholder not in prompt context portion"
        # Evaluator
        gm = compute_masked_generation_metrics(answer_text, context_chunks, frozen_retrieved_ids, GROUND_TRUTH_MRNS)
        print(f"  answer len {len(answer_text)} | recall {gm['generation_recall']:.3f} ({gm['relevant_records_found']}/16) | enum {gm['enumerated_patient_count']} unsupp {gm['unsupported_count']}")
        print(f"  finish_reason={gen['finish_reason']} trunc={gen['truncation']} status={gen['truncation_status']} eval_count={gen['ollama_metadata'].get('eval_count')} done_reason={gen['done_reason']}")
        assert gen["ollama_metadata"], "pilot FAIL: Ollama metadata not captured"
        if "eval_count" in gen["ollama_metadata"]:
            assert gen["output_tokens_is_estimated"] is False, "pilot FAIL: eval_count present but marked estimated"
            assert gen["output_tokens"] == gen["ollama_metadata"]["eval_count"], "pilot FAIL: output_tokens != eval_count"
        else:
            assert gen["output_tokens_is_estimated"] is True
        assert gen["done_reason"] is not None or gen["truncation_status"] == "unknown", "pilot FAIL: done_reason not captured"
        assert gen["truncation_status"] in ("truncated", "not_truncated", "unknown"), f"pilot FAIL: invalid truncation_status"
        # Non-zero recall or confirm evaluator
        if gm["relevant_records_found"] == 0:
            print(f"  WARNING: 0 recall for budget {budget} – checking evaluator behavior (may be valid if model enumerated zero relevant)")
        pilot_runs.append({"budget": budget, "gen": gen, "metrics": gm})

    print("\nPilot PASSED all checks.")
    return {
        "context_audit": audit,
        "relevant_ids_dedup": relevant_ids_dedup,
        "frozen_retrieved_ids": frozen_retrieved_ids,
        "context": context,
        "context_chunks": context_chunks,
        "context_hash": ctx_hash,
        "v2_context_hash": v2_hash,
        "pilot_runs": pilot_runs,
    }


def run_experiment() -> Dict:
    """Run full 10-run filtered experiment."""
    print("=" * 60)
    print("Generation-Filtering Experiment — Controlled (filtered instruction)")
    print("=" * 60)
    print(f"Query: {CANONICAL_QUERY}")
    print(f"Ground truth hypertension MRNs: {len(GROUND_TRUTH_MRNS)}")
    print(f"K: {K}, Budgets: {GENERATION_BUDGETS}, Reps: {REPS_PER_BUDGET}")
    print(f"Filter instruction: {FILTER_INSTRUCTION[:80]}...")

    mrn_records = _load_mrn_records_raw()
    print(f"\nLoaded {len(mrn_records)} MRN records from data/sample_patient_data.txt")
    assert len(mrn_records) == 120, f"Dataset: expected 120 medical records, got {len(mrn_records)}"

    print("\nBuilding medical masked index (pre-embedding masking)...")
    start = time.time()
    vector_store, chunks, chunk_record_map = _build_masked_index(mrn_records)
    elapsed = time.time() - start
    print(f"Index: {len(chunks)} chunks from {len(mrn_records)} records ({elapsed:.2f}s)")

    print(f"\nRetrieving once at K={K} for canonical query...")
    retrieved, _scores = _retrieve_once(vector_store, chunk_record_map, CANONICAL_QUERY, k=K)
    retrieved_ids = [r["record_id"] for r in retrieved]
    seen = set()
    retrieved_ids_dedup = []
    for rid in retrieved_ids:
        if rid not in seen:
            seen.add(rid)
            retrieved_ids_dedup.append(rid)
    relevant_ids_dedup = [rid for rid in retrieved_ids_dedup if rid in GROUND_TRUTH_MRNS]
    retrieved_count = len(retrieved)
    relevant_count = len(relevant_ids_dedup)

    print(f"Retrieved count: {retrieved_count}")
    print(f"Relevant retrieved (dedup): {relevant_count}/{EXPECTED_HYPERTENSION_COUNT}")
    print(f"Retrieved IDs (order): {retrieved_ids_dedup}")

    if retrieved_count != EXPECTED_K or relevant_count != EXPECTED_HYPERTENSION_COUNT:
        raise RuntimeError(
            f"Retrieval verification FAILED: retrieved_count={retrieved_count} (expected {EXPECTED_K}), "
            f"relevant_count={relevant_count} (expected {EXPECTED_HYPERTENSION_COUNT}). "
            f"Missing: {sorted(set(GROUND_TRUTH_MRNS) - set(retrieved_ids_dedup))}"
        )
    print("Retrieval verification PASSED: K=20 retrieved all 16 relevant records.")

    context, context_chunks, frozen_retrieved_ids = _assemble_context(chunks, retrieved)
    ctx_hash = _context_hash(context)
    v2_hash = None
    v2_context = None
    if V2_RESULTS_PATH.exists():
        try:
            v2_data = json.loads(V2_RESULTS_PATH.read_text())
            v2_context = v2_data.get("retrieval", {}).get("context", "")
            if v2_context:
                v2_hash = _context_hash(v2_context)
                print(f"Context hash: filtered={ctx_hash} v2={v2_hash} match={ctx_hash==v2_hash}")
                if ctx_hash != v2_hash:
                    raise RuntimeError(f"Context identity FAILED: filtered hash {ctx_hash} != v2 hash {v2_hash}")
                print("Context identity PASSED: filtered context identical to v2.")
        except Exception as e:
            if "Context identity FAILED" in str(e):
                raise
            print(f"Warning: v2 hash check skipped due to {e}")

    provider = LLM_PROVIDER
    # Build filtered exact prompt for logging (representative 800 budget prompt)
    if provider == "ollama":
        exact_prompt = _build_filtered_ollama_prompt(context, f"{CANONICAL_QUERY}\n\nAnswer:")
        system_prompt = None
        user_prompt = exact_prompt
        messages = None
    else:
        messages = _build_filtered_hf_messages(context, f"{CANONICAL_QUERY}\n\nAnswer:")
        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]
        exact_prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"

    audit = _verify_frozen_context(context, context_chunks)
    assert not audit["context_contains_raw"], f"Raw MRNs leaked into context: {audit['raw_mrns_in_context']}"
    prompt_audit = _verify_prompt_no_raw_mrns(exact_prompt)
    assert not prompt_audit["prompt_contains_raw"], f"Raw MRNs in prompt: {prompt_audit['raw_mrns_in_prompt']}"
    assert prompt_audit["has_filter_instruction"], "Filter instruction missing from prompt"
    print("\nFrozen context assembled.")
    print(f"Context length: {len(context)} chars, {len(context_chunks)} chunks, hash={ctx_hash[:16]}...")
    print(f"Exact filtered prompt length: {len(exact_prompt)} chars")
    print(f"Provider: {provider}, Model: {os.getenv('HF_MODEL' if provider!='ollama' else 'OLLAMA_MODEL', 'default')}, Temperature: {TEMPERATURE}")
    print(f"Context audit: has_masked={audit['has_masked_placeholder']}, raw_mrns_in_context={audit['raw_mrn_count']}")
    print(f"Prompt audit: has_filter={prompt_audit['has_filter_instruction']}, raw_mrns_in_prompt={prompt_audit['raw_mrn_count_prompt']}")

    runs: List[Dict] = []
    run_id_counter = 0
    for budget in GENERATION_BUDGETS:
        for rep in range(1, REPS_PER_BUDGET + 1):
            run_id_counter += 1
            run_id = f"filter_{budget}_rep{rep}"
            print(f"\n[{run_id_counter}/10] Generating {run_id} (budget={budget})...")
            gen = _call_generation_filtered(context, f"{CANONICAL_QUERY}\n\nAnswer:", budget, temperature=TEMPERATURE)
            answer_text = gen["answer_text"]
            gm = compute_masked_generation_metrics(answer_text, context_chunks, frozen_retrieved_ids, GROUND_TRUTH_MRNS)

            run_entry = {
                "run_id": run_id,
                "generation_budget": budget,
                "budget": budget,
                "repetition": rep,
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "query": CANONICAL_QUERY,
                "filter_instruction": FILTER_INSTRUCTION,
                "retrieval_ids": frozen_retrieved_ids,
                "relevant_retrieval_ids": relevant_ids_dedup,
                "retrieval_count": retrieved_count,
                "relevant_retrieval_count": relevant_count,
                "context": context,
                "context_chunks": context_chunks,
                "context_hash": ctx_hash,
                "v2_context_hash": v2_hash,
                "prompt": gen["prompt"],
                "system_prompt": gen["system_prompt"],
                "user_prompt": gen["user_prompt"],
                "messages": gen["messages"],
                "provider": gen["provider"],
                "model": gen["model"],
                "temperature": gen["temperature"],
                "answer_text": answer_text,
                "finish_reason": gen["finish_reason"],
                "done_reason": gen["done_reason"],
                "truncation": gen["truncation"],
                "truncation_status": gen["truncation_status"],
                "output_tokens": gen["output_tokens"],
                "output_tokens_is_estimated": gen["output_tokens_is_estimated"],
                "prompt_tokens": gen["prompt_tokens"],
                "prompt_tokens_is_estimated": gen["prompt_tokens_is_estimated"],
                "latency_ms": gen["latency_ms"],
                "ollama_metadata": gen["ollama_metadata"],
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
                "measurand": gm["measurand"],
            }
            runs.append(run_entry)
            print(f"  -> recall {gm['generation_recall']:.3f} ({gm['relevant_records_found']}/16), missing {gm['missing_count']}, dup {gm['duplicate_records']}, unsupp {gm['unsupported_count']}, enum {gm['enumerated_patient_count']}, trunc {gen['truncation_status']} ({gen['finish_reason']}), tokens {gen['output_tokens']} (est={gen['output_tokens_is_estimated']}), {gen['latency_ms']}ms")

    result = {
        "version": "filtered_v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "description": "Controlled generation-filtering experiment: freeze retrieval/context at K=20 (16/16 relevant), vary only max output tokens [800,1200] x5 with filtered instruction. Measurand is masked-record coverage via deterministic fingerprint matching. Token/truncation via provider Ollama metadata.",
        "query": CANONICAL_QUERY,
        "filter_instruction": FILTER_INSTRUCTION,
        "ground_truth_mrns": GROUND_TRUTH_MRNS,
        "ground_truth_count": len(GROUND_TRUTH_MRNS),
        "retrieval": {
            "k": K,
            "retrieved_count": retrieved_count,
            "relevant_count": relevant_count,
            "retrieved_ids": frozen_retrieved_ids,
            "relevant_retrieval_ids": relevant_ids_dedup,
            "context": context,
            "context_chunks": context_chunks,
            "context_hash": ctx_hash,
            "v2_context_hash": v2_hash,
            "context_identity_match": ctx_hash == v2_hash if v2_hash else None,
            "prompt": exact_prompt,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "provider": provider,
            "model": runs[0]["model"] if runs else None,
            "temperature": TEMPERATURE,
            "context_audit": audit,
            "prompt_audit": prompt_audit,
        },
        "budgets": GENERATION_BUDGETS,
        "reps_per_budget": REPS_PER_BUDGET,
        "total_runs": len(runs),
        "runs": runs,
        "measurand_definition": "enumerated patient/record coverage via deterministic masked-record fingerprint matching (age + diagnosis + treatment + admission) without requiring raw MRN strings; recall = relevant masked records represented in answer / 16; unsupported = non-relevant retrieved records enumerated; duplicate = extra mentions beyond first",
        "instrumentation": "Ollama final metadata captured (done, done_reason, eval_count, prompt_eval_count, total_duration, etc.); output_tokens = eval_count when available (is_estimated=false); truncation = (done_reason == 'length'); finish_reason = exact done_reason ('stop'|'length'|...) or 'unknown' if missing",
    }
    return result


def save_results(result: Dict, path=None) -> Path:
    if path is None:
        path = RESULTS_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def load_results(path=None) -> Dict:
    if path is None:
        path = RESULTS_PATH
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generation-filtering experiment runner")
    parser.add_argument("--output", type=str, default=None, help="Override output path")
    parser.add_argument("--pilot-only", action="store_true", help="Run only 2-run pilot and exit")
    args = parser.parse_args()

    if args.pilot_only:
        pilot = run_pilot()
        print("\nPilot complete.")
    else:
        print("Running mandatory pilot first...\n")
        pilot = run_pilot()
        print("\nPilot passed – proceeding to full 10-run experiment.\n")
        res = run_experiment()
        out = save_results(res, path=args.output)
        print(f"\nResults saved to {out}")
        print(f"Total runs: {res['total_runs']}")
        from .generation_metrics import aggregate_by_budget

        flat = [
            {
                "generation_budget": r["generation_budget"],
                "generation_recall": r["generation_recall"],
                "missing_count": r["missing_count"],
                "duplicate_records": r["duplicate_records"],
                "unsupported_count": r["unsupported_count"],
                "truncation": r["truncation"],
                "truncation_status": r["truncation_status"],
                "latency_ms": r["latency_ms"],
                "output_tokens": r["output_tokens"],
                "prompt_tokens": r.get("prompt_tokens"),
            }
            for r in res["runs"]
        ]
        agg = aggregate_by_budget(flat)
        print("\nAggregate by budget (filtered):")
        print(json.dumps(agg, indent=2))
        metrics_path = METRICS_PATH
        if args.output:
            metrics_path = Path(args.output).parent / "generation_filter_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
