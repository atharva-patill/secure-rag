#!/usr/bin/env python3
"""
Adaptive Retrieval Depth — Runner (research-only)

Evaluates exactly K = [2,5,10,20,30,50] for 5 aggregate queries:
  AGG_HYPERTENSION (16), AGG_AMLODIPINE (17), AGG_PARACETAMOL (20),
  AGG_METFORMIN (7), AGG_T2D_HYPERTENSION (1)

Design:
- Dense FAISS, existing embedding model, existing medical masking,
  existing vector store, existing retrieval implementation.
- Does NOT change production retriever default (K=2).
- Passes K explicitly from research runner.
- Same deterministic retrieval index and query processing.
- Records: query id, query text, K, retrieved chunk indices,
  retrieved record IDs, scores/distances, relevant record IDs,
  intersection, number relevant retrieved.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from benchmarks.retrieval.ground_truth import AGGREGATE_QUERIES_V2
from secure_rag.detection import load_detector_stack
from secure_rag.embedding import embed_chunks
from secure_rag.masker import mask_text
from secure_rag.pdf_loader import chunk_text
from secure_rag.policies import load_policy
from secure_rag.vector_store import VectorStore

RETRIEVAL_DIR = Path(__file__).parent
PROJECT_DIR = Path(__file__).parent.parent.parent.parent
RESULTS_PATH = RETRIEVAL_DIR / "adaptive_k_retrieval_results.json"

# Exactly these K values — no others unless debugging
K_VALUES = [2, 5, 10, 20, 30, 50]

# Exactly these 5 aggregate queries
AGG_QIDS = [
    "AGG_HYPERTENSION",
    "AGG_AMLODIPINE_5MG",
    "AGG_PARACETAMOL_650MG",
    "AGG_METFORMIN_500MG",
    "AGG_T2D_HYPERTENSION",
]

# Ground-truth counts verified programmatically
EXPECTED_COUNTS = {
    "AGG_HYPERTENSION": 16,
    "AGG_AMLODIPINE_5MG": 17,
    "AGG_PARACETAMOL_650MG": 20,
    "AGG_METFORMIN_500MG": 7,
    "AGG_T2D_HYPERTENSION": 1,
}


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
    """Build index using medical pre-embedding masking (production masking path)."""
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
    # VectorStore.search returns (distances, indices) where indices is list
    # and distances is np array converted to list
    if isinstance(indices, list) and len(indices) > 0 and isinstance(indices[0], list):
        indices = indices[0]
    if hasattr(distances, "tolist"):
        distances_list = distances[0].tolist() if hasattr(distances, "shape") and getattr(distances, "ndim", 1) > 1 else distances.tolist()
        if isinstance(distances_list, list) and len(distances_list) > 0 and isinstance(distances_list[0], list):
            distances_list = distances_list[0]
    else:
        distances_list = list(distances)
    # Ensure both are lists
    indices_list = indices if isinstance(indices, list) else list(indices)
    distances_list = distances_list if isinstance(distances_list, list) else list(distances_list)
    return indices_list, distances_list


def verify_ground_truth():
    """Verify ground-truth counts programmatically before running."""
    issues = []
    gt_map = {q["qid"]: q for q in AGGREGATE_QUERIES_V2}
    for qid, expected in EXPECTED_COUNTS.items():
        entry = gt_map.get(qid)
        if entry is None:
            issues.append(f"FAIL: {qid} not found in AGGREGATE_QUERIES_V2")
            continue
        actual = len(entry["relevant_records"])
        if actual != expected:
            issues.append(f"FAIL: {qid} expected {expected}, got {actual}")
        else:
            issues.append(f"PASS: {qid} {actual} records")
    # Also verify we have exactly 5
    if len(AGG_QIDS) != 5:
        issues.append(f"FAIL: expected 5 aggregate QIDs, got {len(AGG_QIDS)}")
    for qid in AGG_QIDS:
        if qid not in gt_map:
            issues.append(f"FAIL: {qid} missing from ground truth")
    return issues


def run_retrieval() -> dict:
    print("=" * 60)
    print("Adaptive K — Retrieval Experiment")
    print("=" * 60)
    print(f"K values: {K_VALUES}")
    print(f"Queries: {AGG_QIDS}")

    # Verify ground truth first
    print("\nVerifying ground truth...")
    for line in verify_ground_truth():
        print(f"  {line}")
    # Hard check
    gt_map = {q["qid"]: q for q in AGGREGATE_QUERIES_V2}
    for qid, expected in EXPECTED_COUNTS.items():
        assert len(gt_map[qid]["relevant_records"]) == expected, f"Ground truth mismatch for {qid}"

    mrn_records = _load_mrn_records_raw()
    print(f"\nLoaded {len(mrn_records)} MRN records")
    assert len(mrn_records) == 120, f"Expected 120 records, got {len(mrn_records)}"

    print("\nBuilding masked index (medical pre-embedding masking)...")
    vector_store, chunks, chunk_record_map = _build_masked_index(mrn_records)
    print(f"Index: {len(chunks)} chunks from {len(mrn_records)} records")
    # Each record is one chunk (since records <500 words)
    # Validate chunk->record map length
    assert len(chunks) == len(chunk_record_map)

    # Prepare queries
    queries = []
    for qid in AGG_QIDS:
        entry = gt_map[qid]
        queries.append({
            "qid": qid,
            "question": entry["question"],
            "relevant_records": sorted(entry["relevant_records"]),
            "expected_behaviour": entry["expected_behaviour"],
        })

    results = []
    for q in queries:
        qid = q["qid"]
        question = q["question"]
        relevant = set(q["relevant_records"])
        print(f"\n[{qid}] {question} (relevant={len(relevant)})")
        per_k = {}
        for k in K_VALUES:
            indices_list, distances_list = _retrieve_top_k(question, vector_store, k=k)
            # Build retrieved record IDs in order
            retrieved_record_ids = []
            retrieved_details = []
            for rank, (chunk_idx, score) in enumerate(zip(indices_list, distances_list)):
                rid = chunk_record_map[chunk_idx] if 0 <= chunk_idx < len(chunk_record_map) else "UNKNOWN"
                retrieved_record_ids.append(rid)
                retrieved_details.append({
                    "rank": rank,
                    "chunk_index": int(chunk_idx),
                    "record_id": rid,
                    "score": round(float(score), 4),
                    "relevant": rid in relevant,
                })
            # Deduplicated relevant retrieved
            seen = set()
            for rid in retrieved_record_ids:
                if rid in relevant:
                    seen.add(rid)
            intersection = sorted(seen)
            num_relevant_retrieved = len(seen)
            # Privacy check: ensure no raw MRN in retrieved chunks text? But retrieved chunks are masked, so shouldn't contain raw MRNs
            # We record for audit but evaluation uses external mapping

            # Verify context integrity later: retrieved records == records included in context (must hold)
            # Ensure K == retrieved count (unless capped at ntotal but ntotal=120 >50 so exact)
            assert len(retrieved_record_ids) == k, f"K mismatch for {qid} k={k}: got {len(retrieved_record_ids)}"

            print(f"  K={k:2d} -> relevant_retrieved {num_relevant_retrieved}/{len(relevant)} recall={num_relevant_retrieved/len(relevant):.3f} precision={num_relevant_retrieved/k:.3f}")

            per_k[str(k)] = {
                "k": k,
                "query_id": qid,
                "query_text": question,
                "retrieved_chunk_indices": [int(x) for x in indices_list],
                "retrieved_record_ids": retrieved_record_ids,
                "retrieved_details": retrieved_details,
                "scores": [round(float(s), 4) for s in distances_list],
                "distances": [round(float(s), 4) for s in distances_list],
                "relevant_record_ids": sorted(relevant),
                "intersection": intersection,
                "num_relevant_retrieved": num_relevant_retrieved,
                "total_relevant": len(relevant),
            }
        results.append({
            "qid": qid,
            "question": question,
            "expected_behaviour": q["expected_behaviour"],
            "relevant_records": sorted(relevant),
            "total_relevant": len(relevant),
            "per_k": per_k,
        })

    output = {
        "version": "adaptive_k_v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "description": "Adaptive Retrieval Depth — retrieval-only evaluation. Dense FAISS, medical masking, K=2,5,10,20,30,50, 5 aggregate queries. Passes K explicitly; production default K=2 unchanged.",
        "k_values": K_VALUES,
        "queries": AGG_QIDS,
        "ground_truth_counts": EXPECTED_COUNTS,
        "total_records": len(mrn_records),
        "num_chunks": len(chunks),
        "index_type": "masked_medical",
        "results": results,
        "ground_truth_verified": True,
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
    parser = argparse.ArgumentParser(description="Adaptive K retrieval runner")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    res = run_retrieval()
    out = save_results(res, path=args.output)
    print(f"\nResults saved to {out}")
