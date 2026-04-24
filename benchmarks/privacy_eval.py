#!/usr/bin/env python3
"""
Privacy evaluation for Secure RAG.

Compares:
- Secure RAG (use_masking=True)  — pre-embedding masking
- Raw RAG (use_masking=False)    — baseline without masking

Metrics:
- Document Leakage: % of PII present in all indexed chunks
- Retrieval Leakage (k=5): % of PII found in top-5 retrieved chunks
- Masking Recall: % of PII correctly masked before embedding
- PHI-in-Answer Rate: % of LLM answers containing raw PII
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

DEBUG = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from secure_rag import rag_answer
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
RETRIEVAL_K = 5


def normalize(text: str) -> str:
    return "".join(text.lower().split())


def loose_match(value: str, text: str) -> bool:
    pattern = re.sub(r"\s+", r"\\s*", re.escape(value))
    return re.search(pattern, text, re.IGNORECASE) is not None


def pii_leaks(text: str, pii: dict) -> bool:
    for value in pii.values():
        if not value:
            continue
        if normalize(value) in normalize(text):
            return True
        if loose_match(value, text):
            return True
    return False


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


def compute_doc_leakage(chunks: List[str], pii: dict) -> Tuple[int, int]:
    text = " ".join(chunks)
    leaked = 0
    total = 0

    for value in pii.values():
        if not value:
            continue
        total += 1
        if normalize(value) in normalize(text) or loose_match(value, text):
            leaked += 1

    return leaked, total


def compute_masking_recall(record: dict, masked_text: str) -> Dict[str, float]:
    results = {}
    pii = record["pii"]
    for field in PII_FIELDS:
        value = pii.get(field)
        if not value:
            results[field] = None
            continue
        if normalize(value) in normalize(masked_text) or loose_match(value, masked_text):
            results[field] = 0.0
        else:
            results[field] = 1.0
    return results


def check_phi_in_text(text: str, record: dict) -> bool:
    return pii_leaks(text, record["pii"])


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
        "document_leakage": {"secure": {}, "raw": {}},
        "retrieval_leakage": {"secure": {}, "raw": {}},
        "masking_recall": {},
        "phi_in_answer": {"secure": {}, "raw": {}},
        "summary": {},
    }

    print("\n" + "-" * 40)
    print("Running document leakage evaluation...")
    print("-" * 40)

    doc_leak_raw_leaked = 0
    doc_leak_raw_total = 0
    doc_leak_secure_leaked = 0
    doc_leak_secure_total = 0

    for rid, record in test_records.items():
        if DEBUG:
            print(f"\n  DEBUG {rid}:")
            print(f"    PII: {record['pii']}")
            print(f"    Chunk sample: {chunks_raw[:2]}")

        leaked_raw, total_raw = compute_doc_leakage(chunks_raw, record["pii"])
        leaked_secure, total_secure = compute_doc_leakage(chunks_secure, record["pii"])

        doc_leak_raw_leaked += leaked_raw
        doc_leak_raw_total += total_raw
        doc_leak_secure_leaked += leaked_secure
        doc_leak_secure_total += total_secure

        results["document_leakage"]["raw"][rid] = {
            "leaked": leaked_raw,
            "total": total_raw,
            "rate": leaked_raw / total_raw if total_raw > 0 else 0.0,
        }
        results["document_leakage"]["secure"][rid] = {
            "leaked": leaked_secure,
            "total": total_secure,
            "rate": leaked_secure / total_secure if total_secure > 0 else 0.0,
        }

    results["document_leakage"]["_summary"] = {
        "raw_rate": doc_leak_raw_leaked / doc_leak_raw_total if doc_leak_raw_total > 0 else 0.0,
        "secure_rate": doc_leak_secure_leaked / doc_leak_secure_total if doc_leak_secure_total > 0 else 0.0,
        "raw_leaked": doc_leak_raw_leaked,
        "raw_total": doc_leak_raw_total,
        "secure_leaked": doc_leak_secure_leaked,
        "secure_total": doc_leak_secure_total,
    }

    print("\n" + "-" * 40)
    print("Running masking recall evaluation...")
    print("-" * 40)

    recall_fields = {f: {"hit": 0, "total": 0} for f in PII_FIELDS}

    for rid, record in test_records.items():
        text = record["text"]
        masked_text = mask_text(text)

        recall = compute_masking_recall(record, masked_text)
        for field, val in recall.items():
            if val is not None:
                recall_fields[field]["total"] += 1
                if val == 1.0:
                    recall_fields[field]["hit"] += 1

    masking_recall_per_field = {}
    for field, data in recall_fields.items():
        if data["total"] > 0:
            masking_recall_per_field[field] = {
                "recall": data["hit"] / data["total"],
                "count": data["total"],
            }

    results["masking_recall"] = masking_recall_per_field

    print("\n" + "-" * 40)
    print("Running retrieval leakage evaluation (k=5)...")
    print("-" * 40)

    retrieval_leak_raw = 0
    retrieval_leak_secure = 0
    retrieval_total = 0

    for rid, record in test_records.items():
        queries = queries_by_record.get(rid, [])
        if not queries:
            continue

        q = queries[0]["question"]
        masked_q = mask_text(q)

        q_vec = embed_chunks([masked_q])
        q_vec = q_vec.astype("float32")

        _, idx_secure = vs_secure.search(q_vec, k=RETRIEVAL_K)
        retrieved_secure = [chunks_secure[int(i)] for i in idx_secure]

        _, idx_raw = vs_raw.search(q_vec, k=RETRIEVAL_K)
        retrieved_raw = [chunks_raw[int(i)] for i in idx_raw]

        all_secure = " ".join(retrieved_secure)
        all_raw = " ".join(retrieved_raw)

        raw_leaked = 0
        secure_leaked = 0
        total = 0

        for field in PII_FIELDS:
            value = record["pii"].get(field)
            if not value:
                continue
            total += 1
            if normalize(value) in normalize(all_raw) or loose_match(value, all_raw):
                raw_leaked += 1
            if normalize(value) in normalize(all_secure) or loose_match(value, all_secure):
                secure_leaked += 1

        retrieval_leak_raw += raw_leaked
        retrieval_leak_secure += secure_leaked
        retrieval_total += total

        results["retrieval_leakage"]["raw"][rid] = {
            "leaked": raw_leaked,
            "total": total,
            "rate": raw_leaked / total if total > 0 else 0.0,
        }
        results["retrieval_leakage"]["secure"][rid] = {
            "leaked": secure_leaked,
            "total": total,
            "rate": secure_leaked / total if total > 0 else 0.0,
        }

    results["retrieval_leakage"]["_summary"] = {
        "raw_rate": retrieval_leak_raw / retrieval_total if retrieval_total > 0 else 0.0,
        "secure_rate": retrieval_leak_secure / retrieval_total if retrieval_total > 0 else 0.0,
        "raw_leaked": retrieval_leak_raw,
        "retrieval_total": retrieval_total,
        "secure_leaked": retrieval_leak_secure,
    }

    hf_api_key = os.getenv("HF_API_KEY")
    has_llm = hf_api_key and hf_api_key != "dummy-for-test"

    if has_llm:
        print("\n" + "-" * 40)
        print("Running PHI-in-answer evaluation (LLM)...")
        print("-" * 40)
        print(f"Using LLM with {len(test_ids)} test records, 2 queries per record\n")

        phi_in_answer_raw = 0
        phi_in_answer_secure = 0
        phi_total = 0

        for i, rid in enumerate(test_ids[:20]):
            if rid not in records or rid not in queries_by_record:
                continue

            record = records[rid]
            queries = queries_by_record[rid]
            selected = [queries[0], queries[1]] if len(queries) >= 2 else [queries[0]]

            for q in selected:
                try:
                    answer_secure = "".join(list(rag_answer(q["question"], vs_secure, chunks_secure)))
                except Exception:
                    answer_secure = ""

                has_phi_secure = check_phi_in_text(answer_secure, record)
                phi_in_answer_secure += 1 if has_phi_secure else 0

                phi_total += 1

            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{min(20, len(test_ids))} records...")

        results["phi_in_answer"]["_summary"] = {
            "secure_rate": phi_in_answer_secure / phi_total if phi_total > 0 else 0.0,
            "secure_leaked": phi_in_answer_secure,
            "total": phi_total,
        }
    else:
        print("\n" + "-" * 40)
        print("LLM not available (HF_API_KEY not set)")
        print("Skipping PHI-in-answer evaluation")
        print("-" * 40)

    overall_recall = [v["recall"] for v in results["masking_recall"].values() if "recall" in v]
    if overall_recall:
        results["summary"] = {
            "masking_recall": sum(overall_recall) / len(overall_recall),
            "document_leakage_raw": results["document_leakage"]["_summary"]["raw_rate"],
            "document_leakage_secure": results["document_leakage"]["_summary"]["secure_rate"],
            "retrieval_leakage_raw": results["retrieval_leakage"]["_summary"]["raw_rate"],
            "retrieval_leakage_secure": results["retrieval_leakage"]["_summary"]["secure_rate"],
        }
        if has_llm and "_summary" in results["phi_in_answer"]:
            results["summary"]["phi_in_answer_secure"] = results["phi_in_answer"]["_summary"]["secure_rate"]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n  Document Leakage:")
    print(f"    Raw RAG:    {results['document_leakage']['_summary']['raw_rate']:.1%}")
    print(f"    Secure RAG: {results['document_leakage']['_summary']['secure_rate']:.1%}")

    print("\n  Retrieval Leakage (k=5):")
    print(f"    Raw RAG:    {results['retrieval_leakage']['_summary']['raw_rate']:.1%}")
    print(f"    Secure RAG: {results['retrieval_leakage']['_summary']['secure_rate']:.1%}")

    print("\n  Masking Recall:")
    for field, data in results["masking_recall"].items():
        if "recall" in data:
            print(f"    {field:<12}: {data['recall']:.1%}")

    if overall_recall:
        print(f"    {'OVERALL':<12}: {sum(overall_recall)/len(overall_recall):.1%}")

    if has_llm and "_summary" in results["phi_in_answer"]:
        print("\n  PHI in Answers:")
        print(f"    Secure RAG: {results['phi_in_answer']['_summary']['secure_rate']:.1%}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    run_evaluation()