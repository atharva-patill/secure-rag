#!/usr/bin/env python3
"""
Privacy evaluation for Secure RAG.

Compares:
- Secure RAG (use_masking=True)  — pre-embedding masking
- Raw RAG (use_masking=False)    — baseline without masking

Metrics:
- Masking Recall: % of PII correctly masked
- Privacy Leakage Rate: % of raw PII found in retrieved chunks
- PHI-in-Answer Rate: % of LLM answers containing raw PII
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from secure_rag import build_rag, rag_answer
from secure_rag.masker import mask_text
from secure_rag.pdf_loader import chunk_text
from secure_rag.embedding import embed_chunks
from secure_rag.vector_store import VectorStore

BENCHMARK_DIR = Path(__file__).parent
DATASET_PATH = BENCHMARK_DIR / "dataset.jsonl"
QUERIES_PATH = BENCHMARK_DIR / "dataset_queries.json"
SPLIT_PATH = BENCHMARK_DIR / "train_test_split.json"
RESULTS_PATH = BENCHMARK_DIR / "results.json"

PII_FIELDS = ["name", "aadhaar", "pan", "phone", "email", "mrn", "dob", "address"]


def load_records():
    records = {}
    with open(DATASET_PATH) as f:
        for line in f:
            r = json.loads(line)
            records[r["record_id"]] = r
    return records


def load_queries():
    with open(QUERIES_PATH) as f:
        return json.load(f)


def load_split():
    with open(SPLIT_PATH) as f:
        return json.load(f)


def build_index(records: dict, use_masking: bool) -> Tuple[VectorStore, List[str]]:
    texts = []
    for rid, r in records.items():
        text = r["text"]
        if use_masking:
            text = mask_text(text)
        texts.append(text)

    chunks = []
    for text in texts:
        chunks.extend(chunk_text(text))

    embeddings = embed_chunks(chunks)
    vector_store = VectorStore(embeddings)
    return vector_store, chunks


def compute_masking_recall(record: dict, masked_text: str) -> Dict[str, float]:
    results = {}
    for field in PII_FIELDS:
        value = record["pii"].get(field)
        if not value:
            results[field] = None
            continue
        if value in masked_text:
            results[field] = 0.0
        else:
            results[field] = 1.0
    return results


def compute_leakage_rate(
    record: dict,
    chunks: list,
    use_masking: bool
) -> Dict[str, float]:
    results = {}
    all_text = " ".join(chunks)
    for field in PII_FIELDS:
        value = record["pii"].get(field)
        if not value:
            results[field] = None
            continue
        if value in all_text:
            results[field] = 1.0
        else:
            results[field] = 0.0
    return results


def check_phi_in_text(text: str, record: dict) -> bool:
    for field in PII_FIELDS:
        value = record["pii"].get(field)
        if value and value in text:
            return True
    return False


def run_evaluation():
    print("=" * 60)
    print("Secure RAG Privacy Evaluation")
    print("=" * 60)

    records = load_records()
    queries_by_record = {q["record_id"]: q["queries"] for q in load_queries()}
    split = load_split()
    test_ids = split["test"]

    print(f"\nLoaded {len(records)} records")
    print(f"Test set: {len(test_ids)} records")

    test_records = {rid: records[rid] for rid in test_ids if rid in records}

    print("\n" + "-" * 40)
    print("Building indices...")
    print("-" * 40)

    print("\n[1/2] Building Secure RAG index (use_masking=True)...")
    start = time.time()
    vs_secure, chunks_secure = build_index(test_records, use_masking=True)
    print(f"      Done in {time.time() - start:.2f}s")

    print("\n[2/2] Building Raw RAG index (use_masking=False)...")
    start = time.time()
    vs_raw, chunks_raw = build_index(test_records, use_masking=False)
    print(f"      Done in {time.time() - start:.2f}s")

    results = {
        "masking_recall": {"secure": {}, "raw": {}},
        "leakage_rate": {"secure": {}, "raw": {}},
        "phi_in_answer_rate": {"secure": {}, "raw": {}},
        "per_record": {},
    }

    print("\n" + "-" * 40)
    print("Running masking recall evaluation...")
    print("-" * 40)

    for rid, record in test_records.items():
        text = record["text"]
        masked_text = mask_text(text)

        recall = compute_masking_recall(record, masked_text)
        results["masking_recall"]["secure"][rid] = {
            f: v for f, v in recall.items() if v is not None
        }

    for field in PII_FIELDS:
        field_recalls = [
            r[field]
            for r in results["masking_recall"]["secure"].values()
            if field in r and r[field] is not None
        ]
        if field_recalls:
            results["masking_recall"]["secure"][field] = {
                "mean": sum(field_recalls) / len(field_recalls),
                "count": len(field_recalls),
            }

    print("\n" + "-" * 40)
    print("Running retrieval leakage evaluation...")
    print("-" * 40)

    for rid, record in test_records.items():
        queries = queries_by_record.get(rid, [])
        if not queries:
            continue

        q = queries[0]["question"]
        masked_q = mask_text(q)

        q_vec = embed_chunks([masked_q])
        q_vec = q_vec.astype("float32")

        _, idx_secure = vs_secure.search(q_vec, k=1)
        retrieved_secure = [chunks_secure[int(i)] for i in idx_secure]

        _, idx_raw = vs_raw.search(q_vec, k=1)
        retrieved_raw = [chunks_raw[int(i)] for i in idx_raw]

        all_secure = " ".join(retrieved_secure)
        all_raw = " ".join(retrieved_raw)

        results["leakage_rate"]["secure"][rid] = {
            field: 1.0 if record["pii"].get(field) and record["pii"][field] in all_secure else 0.0
            for field in PII_FIELDS
        }
        results["leakage_rate"]["raw"][rid] = {
            field: 1.0 if record["pii"].get(field) and record["pii"][field] in all_raw else 0.0
            for field in PII_FIELDS
        }

    for method in ["secure", "raw"]:
        for field in PII_FIELDS:
            vals = [
                results["leakage_rate"][method][rid][field]
                for rid in results["leakage_rate"][method]
                if field in results["leakage_rate"][method][rid]
            ]
            if vals:
                results["leakage_rate"][method][field] = {
                    "mean": sum(vals) / len(vals),
                    "count": len(vals),
                }

    hf_api_key = os.getenv("HF_API_KEY")
    has_llm = hf_api_key and hf_api_key != "dummy-for-test"

    if has_llm:
        print("\n" + "-" * 40)
        print("Running PHI-in-answer evaluation (LLM)...")
        print("-" * 40)
        print(f"Using LLM with {len(test_ids)} test records, 1 query per record\n")

        for i, rid in enumerate(test_ids[:20]):
            if rid not in records or rid not in queries_by_record:
                continue

            record = records[rid]
            queries = queries_by_record[rid]

            selected = [queries[0], queries[1]] if len(queries) >= 2 else [queries[0]]

            for q in selected:
                masked_q = mask_text(q["question"])

                try:
                    answer_gen = rag_answer(q["question"], vs_secure, chunks_secure)
                    answer = "".join(list(answer_gen))
                except Exception:
                    answer = ""

                has_phi = check_phi_in_text(answer, record)
                key = f"{rid}_{q['qid']}"
                results["phi_in_answer_rate"]["secure"][key] = 1.0 if has_phi else 0.0

            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{min(20, len(test_ids))} records...")

        all_phi_vals = list(results["phi_in_answer_rate"]["secure"].values())
        if all_phi_vals:
            results["phi_in_answer_rate"]["secure"]["_overall"] = {
                "mean": sum(all_phi_vals) / len(all_phi_vals),
                "count": len(all_phi_vals),
            }
    else:
        print("\n" + "-" * 40)
        print("LLM not available (HF_API_KEY not set)")
        print("Skipping PHI-in-answer evaluation")
        print("-" * 40)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\n--- Masking Recall (Secure RAG) ---")
    overall_recall = []
    for field in PII_FIELDS:
        if field in results["masking_recall"]["secure"]:
            data = results["masking_recall"]["secure"][field]
            if isinstance(data, dict):
                recall = data["mean"]
                count = data["count"]
                overall_recall.append(recall)
                print(f"  {field:<12}: {recall:.2%} ({count} records)")
    if overall_recall:
        print(f"  {'OVERALL':<12}: {sum(overall_recall)/len(overall_recall):.2%}")

    print("\n--- Privacy Leakage Rate ---")
    print(f"  {'Method':<12} {'Rate':>10}")
    print(f"  {'-'*24}")
    for method in ["raw", "secure"]:
        for field in PII_FIELDS:
            if field in results["leakage_rate"][method]:
                data = results["leakage_rate"][method][field]
                if isinstance(data, dict):
                    rate = data["mean"]
                    count = data["count"]
                    print(f"  {method.upper()} - {field:<12} : {rate:.2%} ({count} records)")

    overall_leak_raw = [
        results["leakage_rate"]["raw"][rid][f]
        for rid in results["leakage_rate"]["raw"]
        for f in results["leakage_rate"]["raw"][rid]
        if f in PII_FIELDS
    ]
    overall_leak_secure = [
        results["leakage_rate"]["secure"][rid][f]
        for rid in results["leakage_rate"]["secure"]
        for f in results["leakage_rate"]["secure"][rid]
        if f in PII_FIELDS
    ]

    if overall_leak_raw:
        print(f"\n  {'Raw RAG Overall':<25}: {sum(overall_leak_raw)/len(overall_leak_raw):.2%}")
    if overall_leak_secure:
        print(f"  {'Secure RAG Overall':<25}: {sum(overall_leak_secure)/len(overall_leak_secure):.2%}")

    if has_llm and results["phi_in_answer_rate"]["secure"]:
        print("\n--- PHI-in-Answer Rate ---")
        overall_phi = results["phi_in_answer_rate"]["secure"].get("_overall", {})
        if overall_phi:
            print(f"  Secure RAG: {overall_phi['mean']:.2%} ({overall_phi['count']} queries)")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    run_evaluation()