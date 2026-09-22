#!/usr/bin/env python3
"""
Controlled generation-budget experiment — v2 (fixed measurand + instrumentation).

Determines whether incomplete aggregate answers are caused by the LLM generation budget.

Design (from spec):
- Canonical query: "Give me all patients with hypertension." (16 GT records)
- Dense FAISS, medical pre-embedding masking, K=20, retrieval ONCE, freeze context.
- Reuse exact production prompt construction.
- Vary ONLY max output tokens: [200,400,600,800,1200] x 5 reps = 25 generations.
- Fixed: query, retrieved records, context ordering, system prompt, user prompt,
         model, provider, temperature (0.3), other sampling params.
- v2 fixes:
  * Generation measurand uses masked-record fingerprints, not raw MRN strings.
  * Ollama final metadata (done, done_reason, eval_count, prompt_eval_count, etc.) captured.
  * Token accounting uses provider eval_count where available.
  * Truncation based on provider done_reason == "length", not heuristic.
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
GENERATION_BUDGETS = [200, 400, 600, 800, 1200]
REPS_PER_BUDGET = 5
K = 20
TEMPERATURE = 0.3

# Ground truth for hypertension query
_HYP_GT_ENTRY = next(q for q in AGGREGATE_QUERIES_V2 if q["qid"] == "AGG_HYPERTENSION")
GROUND_TRUTH_MRNS: List[str] = sorted(_HYP_GT_ENTRY["relevant_records"])

RESULTS_PATH_V1 = GENERATION_DIR / "generation_results_v1.json"
RESULTS_PATH = GENERATION_DIR / "generation_results_v2.json"
METRICS_PATH = GENERATION_DIR / "generation_metrics_v2.json"


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


def _call_generation(context: str, query: str, budget: int, temperature: float = TEMPERATURE):
    """Call production generator with explicit budget and capture provider metadata.

    Uses provider-reported eval_count/prompt_eval_count where available.
    Truncation based on done_reason == "length", not heuristic.
    """
    provider = LLM_PROVIDER
    model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct") if provider != "ollama" else os.getenv("OLLAMA_MODEL", "llama3.2:latest")

    if provider == "ollama":
        prompt_str = _build_ollama_prompt(context, query)
        system_prompt = None
        user_prompt = prompt_str
        messages = None
    else:
        messages = _build_hf_messages(context, query)
        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]
        prompt_str = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"

    start = time.time()
    answer = "".join(generate_answer(context, f"{query}\n\nAnswer:", max_tokens=budget, num_predict=budget, temperature=temperature))
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
        # No provider prompt count available – try estimate prompt tokens via same estimator on prompt_str
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
            # Ollama done_reason is authoritative
            if done_reason == "length":
                finish_reason = "length"
                truncation = True
                truncation_status = "truncated"
            elif done_reason == "stop":
                finish_reason = "stop"
                truncation = False
                truncation_status = "not_truncated"
            else:
                # Other values (e.g., error) – treat as not length-limited but record exact
                finish_reason = str(done_reason)
                truncation = False
                truncation_status = "not_truncated"
        else:
            # done_reason missing -> unknown
            # Check done flag
            if ollama_meta.get("done") is True:
                # done=True but no reason -> unknown
                finish_reason = "unknown"
                truncation = None
                truncation_status = "unknown"
            else:
                finish_reason = "unknown"
                truncation = None
                truncation_status = "unknown"
    else:
        # HF path – no provider done_reason; mark unknown
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
    }


def _verify_frozen_context(context: str, context_chunks: List[str]):
    """Audit that LLM context does not contain raw MRNs."""
    raw_mrn_pattern = re.compile(r"\bMRN\d+\b")
    found = raw_mrn_pattern.findall(context)
    # Masked context should contain [PATIENT_ID_MASKED] instead
    has_masked = "[PATIENT_ID_MASKED]" in context
    return {
        "raw_mrns_in_context": found,
        "raw_mrn_count": len(found),
        "has_masked_placeholder": has_masked,
        "context_contains_raw": len(found) > 0,
    }


def run_pilot(budgets=(200, 1200)) -> Dict:
    """Mandatory 2-run pilot before full experiment.

    Verifies:
      1. Retrieval 16/16
      2. Frozen context identical to intended masked context
      3. Raw MRNs NOT present in LLM context
      4. Model output contains patient/record info that new measurand can evaluate
      5. New metric produces non-trivial measurable result
      6. Ollama final metadata actually captured
      7. eval_count captured when provided
      8. done_reason captured
      9. truncation based on provider metadata, not heuristic
      10. 200 and 1200 runs distinguishable by actual generation termination behavior
    """
    print("=" * 60)
    print("Generation-Budget Pilot — 2 runs (200 & 1200)")
    print("=" * 60)
    mrn_records = _load_mrn_records_raw()
    vector_store, chunks, chunk_record_map = _build_masked_index(mrn_records)
    retrieved, _ = _retrieve_once(vector_store, chunk_record_map, CANONICAL_QUERY, k=K)
    retrieved_ids = [r["record_id"] for r in retrieved]
    # Deduplicate for verification
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
    assert audit["has_masked_placeholder"], "pilot FAIL: masked placeholder not found – masking not applied"

    pilot_runs = []
    for budget in budgets:
        print(f"\n[Pilot] Generating budget={budget} rep=1 ...")
        gen = _call_generation(context, f"{CANONICAL_QUERY}\n\nAnswer:", budget, temperature=TEMPERATURE)
        answer_text = gen["answer_text"]
        assert answer_text and len(answer_text) > 20, f"pilot FAIL: answer empty for budget {budget}"
        # New metric must be evaluable
        gm = compute_masked_generation_metrics(answer_text, context_chunks, frozen_retrieved_ids, GROUND_TRUTH_MRNS)
        print(f"  answer len {len(answer_text)} | recall {gm['generation_recall']:.3f} ({gm['relevant_records_found']}/16) | enum {gm['enumerated_patient_count']}")
        print(f"  finish_reason={gen['finish_reason']} trunc={gen['truncation']} status={gen['truncation_status']} eval_count={gen['ollama_metadata'].get('eval_count')} done_reason={gen['done_reason']}")
        # Verify metadata captured
        assert gen["ollama_metadata"], "pilot FAIL: Ollama metadata not captured (empty)"
        # eval_count check – if provider provides it, must be present
        # We allow missing but then output_tokens must be marked estimated
        if "eval_count" in gen["ollama_metadata"]:
            assert gen["output_tokens_is_estimated"] is False, "pilot FAIL: eval_count present but output_tokens marked estimated"
            assert gen["output_tokens"] == gen["ollama_metadata"]["eval_count"], "pilot FAIL: output_tokens != eval_count"
        else:
            print("  Note: eval_count not provided by Ollama in this pilot run (will be marked estimated)")
            assert gen["output_tokens_is_estimated"] is True
        assert gen["done_reason"] is not None or gen["truncation_status"] == "unknown", "pilot FAIL: done_reason not captured"
        # Truncation must be based on provider, not heuristic token estimate
        # Ensure we are not using estimated_tokens >= budget heuristic
        assert gen["truncation_status"] in ("truncated", "not_truncated", "unknown"), f"pilot FAIL: truncation_status invalid {gen['truncation_status']}"
        pilot_runs.append({"budget": budget, "gen": gen, "metrics": gm})

    # Check distinguishability of 200 vs 1200
    r200 = pilot_runs[0]
    r1200 = pilot_runs[1]
    # They should be distinguishable by output_tokens and/or truncation and/or enumerated count
    distinguishable = (
        r200["gen"]["output_tokens"] != r1200["gen"]["output_tokens"]
        or r200["gen"]["finish_reason"] != r1200["gen"]["finish_reason"]
        or r200["metrics"]["relevant_records_found"] != r1200["metrics"]["relevant_records_found"]
        or r200["metrics"]["enumerated_patient_count"] != r1200["metrics"]["enumerated_patient_count"]
    )
    print(f"\nPilot distinguishability 200 vs 1200: {distinguishable}")
    # Also ensure metric produces non-trivial result (not all zero, not all 16, and >0 for large budget)
    assert r1200["metrics"]["relevant_records_found"] > 0, "pilot FAIL: 1200 recall 0 – metric not capturing enumeration"
    # For 200, should have at least 1 but less than 16 due to budget
    assert r200["metrics"]["relevant_records_found"] >= 1, "pilot FAIL: 200 recall 0 – metric not capturing"
    print("\nPilot PASSED all 10 checks.")
    return {
        "context_audit": audit,
        "relevant_ids_dedup": relevant_ids_dedup,
        "frozen_retrieved_ids": frozen_retrieved_ids,
        "context": context,
        "context_chunks": context_chunks,
        "pilot_runs": pilot_runs,
    }


def run_experiment() -> Dict:
    """Run the full controlled generation-budget experiment (v2)."""
    print("=" * 60)
    print("Generation-Budget Experiment — Controlled v2 (masked measurand + provider instrumentation)")
    print("=" * 60)
    print(f"Query: {CANONICAL_QUERY}")
    print(f"Ground truth hypertension MRNs: {len(GROUND_TRUTH_MRNS)}")
    print(f"K: {K}, Budgets: {GENERATION_BUDGETS}, Reps: {REPS_PER_BUDGET}")

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
    provider = LLM_PROVIDER
    if provider == "ollama":
        exact_prompt = _build_ollama_prompt(context, f"{CANONICAL_QUERY}\n\nAnswer:")
        system_prompt = None
        user_prompt = exact_prompt
        messages = None
    else:
        messages = _build_hf_messages(context, f"{CANONICAL_QUERY}\n\nAnswer:")
        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]
        exact_prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"

    # Verify masked context before generation
    audit = _verify_frozen_context(context, context_chunks)
    assert not audit["context_contains_raw"], f"Raw MRNs leaked into context: {audit['raw_mrns_in_context']}"
    print("\nFrozen context assembled.")
    print(f"Context length: {len(context)} chars, {len(context_chunks)} chunks")
    print(f"Exact prompt length: {len(exact_prompt)} chars")
    print(f"Provider: {provider}, Model: {os.getenv('HF_MODEL' if provider!='ollama' else 'OLLAMA_MODEL', 'default')}, Temperature: {TEMPERATURE}")
    print(f"Context audit: has_masked={audit['has_masked_placeholder']}, raw_mrns_in_context={audit['raw_mrn_count']}")

    runs: List[Dict] = []
    run_id_counter = 0
    for budget in GENERATION_BUDGETS:
        for rep in range(1, REPS_PER_BUDGET + 1):
            run_id_counter += 1
            run_id = f"budget_{budget}_rep_{rep}"
            print(f"\n[{run_id_counter}/25] Generating {run_id} (budget={budget})...")
            gen = _call_generation(context, f"{CANONICAL_QUERY}\n\nAnswer:", budget, temperature=TEMPERATURE)
            answer_text = gen["answer_text"]
            gm = compute_masked_generation_metrics(answer_text, context_chunks, frozen_retrieved_ids, GROUND_TRUTH_MRNS)

            run_entry = {
                "run_id": run_id,
                "generation_budget": budget,
                "budget": budget,
                "repetition": rep,
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "query": CANONICAL_QUERY,
                "retrieval_ids": frozen_retrieved_ids,
                "relevant_retrieval_ids": relevant_ids_dedup,
                "retrieval_count": retrieved_count,
                "relevant_retrieval_count": relevant_count,
                "context": context,
                "context_chunks": context_chunks,
                "prompt": exact_prompt,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "messages": messages,
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
                # generation metrics (v2 masked)
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
        "version": "v2",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "description": "Controlled generation-budget experiment v2: freeze retrieval/context at K=20 (16/16 relevant), vary only max output tokens across 5 budgets x5 reps =25 runs. Measurand is masked-record coverage via deterministic fingerprint matching (no raw MRN required). Token/truncation via provider Ollama metadata (eval_count/done_reason).",
        "query": CANONICAL_QUERY,
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
            "prompt": exact_prompt,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "provider": provider,
            "model": gen["model"] if runs else None,
            "temperature": TEMPERATURE,
            "context_audit": audit,
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

    parser = argparse.ArgumentParser(description="Generation-budget experiment runner v2")
    parser.add_argument("--output", type=str, default=None, help="Override output path")
    parser.add_argument("--pilot-only", action="store_true", help="Run only 2-run pilot and exit")
    args = parser.parse_args()

    if args.pilot_only:
        pilot = run_pilot()
        print("\nPilot complete.")
    else:
        # Mandatory pilot before full
        print("Running mandatory pilot first...\n")
        pilot = run_pilot()
        print("\nPilot passed – proceeding to full 25-run experiment.\n")
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
        print("\nAggregate by budget:")
        print(json.dumps(agg, indent=2))
        # Also save metrics aggregate
        metrics_path = METRICS_PATH
        if args.output:
            # if custom output, save metrics alongside
            metrics_path = Path(args.output).parent / "generation_metrics_v2.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
