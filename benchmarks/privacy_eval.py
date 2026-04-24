#!/usr/bin/env python3
"""
Privacy evaluation for Secure RAG - 3-mode comparison.

Compares three masking strategies:
- raw:   No masking anywhere
- post:  Mask only retrieved context before LLM
- pre:   Mask before embedding (our method)

Metrics:
- Document Leakage: % of PII present in all indexed chunks
- Retrieval Leakage (k=5): % of PII found in top-5 retrieved chunks
- PHI-in-Answer Rate: % of LLM answers containing raw PII
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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

DEBUG = False


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
    print("Secure RAG Privacy Evaluation - 3 Mode Comparison")
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

    print("\n[1/2] Building raw index (use_masking=False)...")
    start = time.time()
    raw_index, raw_chunks = build_index(test_records, use_masking=False)
    print(f"      Done in {time.time() - start:.2f}s")

    print("\n[2/2] Building pre index (use_masking=True)...")
    start = time.time()
    pre_index, pre_chunks = build_index(test_records, use_masking=True)
    print(f"      Done in {time.time() - start:.2f}s")

    results = {
        "document_leakage": {},
        "retrieval_leakage": {},
        "masking_recall": {},
        "phi_in_answer": {},
        "summary": {},
    }

    print("\n" + "-" * 40)
    print("Running document leakage evaluation...")
    print("-" * 40)

    for mode in ["raw", "post", "pre"]:
        chunks = pre_chunks if mode == "pre" else raw_chunks
        results["document_leakage"][mode] = {}

        total_leaked = 0
        total_pii = 0

        for rid, record in test_records.items():
            leaked, total = compute_doc_leakage(chunks, record["pii"])
            total_leaked += leaked
            total_pii += total

            results["document_leakage"][mode][rid] = {
                "leaked": leaked,
                "total": total,
            }

        results["document_leakage"][mode]["_summary"] = {
            "leaked": total_leaked,
            "total": total_pii,
            "rate": total_leaked / total_pii if total_pii > 0 else 0.0,
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

    for mode in ["raw", "post", "pre"]:
        index = pre_index if mode == "pre" else raw_index
        chunks = pre_chunks if mode == "pre" else raw_chunks
        results["retrieval_leakage"][mode] = {}

        total_leaked = 0
        total_pii = 0

        for rid, record in test_records.items():
            queries = queries_by_record.get(rid, [])
            if not queries:
                continue

            q = queries[0]["question"]

            q_vec = embed_chunks([q])
            q_vec = q_vec.astype("float32")

            _, idx = index.search(q_vec, k=RETRIEVAL_K)
            retrieved = [chunks[int(i)] for i in idx]
            all_text = " ".join(retrieved)

            leaked = 0
            total = 0
            for field in PII_FIELDS:
                value = record["pii"].get(field)
                if not value:
                    continue
                total += 1
                if normalize(value) in normalize(all_text) or loose_match(value, all_text):
                    leaked += 1

            total_leaked += leaked
            total_pii += total

            results["retrieval_leakage"][mode][rid] = {
                "leaked": leaked,
                "total": total,
            }

        results["retrieval_leakage"][mode]["_summary"] = {
            "leaked": total_leaked,
            "total": total_pii,
            "rate": total_leaked / total_pii if total_pii > 0 else 0.0,
        }

    hf_api_key = os.getenv("HF_API_KEY")
    has_llm = hf_api_key and hf_api_key != "dummy-for-test"

    if has_llm:
        print("\n" + "-" * 40)
        print("Running PHI-in-answer evaluation (LLM)...")
        print("-" * 40)
        print(f"Using LLM with {len(test_ids)} test records, 2 queries per record\n")

        for mode in ["raw", "post", "pre"]:
            index = pre_index if mode == "pre" else raw_index
            chunks = pre_chunks if mode == "pre" else raw_chunks
            results["phi_in_answer"][mode] = {}

            leaked = 0
            total = 0

            for i, rid in enumerate(test_ids[:20]):
                if rid not in records or rid not in queries_by_record:
                    continue

                record = records[rid]
                queries = queries_by_record[rid]
                phi_queries = [q for q in queries if q.get("phi_in_answer")]
                selected = phi_queries[:3] if phi_queries else (queries[:2] if len(queries) >= 2 else queries[:1])

                for q in selected:
                    try:
                        answer_gen = rag_answer(
                            q["question"],
                            index,
                            chunks,
                            mask_mode=mode
                        )
                        answer = "".join(list(answer_gen))
                    except Exception:
                        answer = ""

                    has_phi = check_phi_in_text(answer, record)
                    leaked += 1 if has_phi else 0
                    total += 1

                if (i + 1) % 5 == 0:
                    print(f"  {mode}: Processed {i + 1}/{min(20, len(test_ids))}...")

            results["phi_in_answer"][mode]["_summary"] = {
                "leaked": leaked,
                "total": total,
                "rate": leaked / total if total > 0 else 0.0,
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
            "document_leakage_raw": results["document_leakage"]["raw"]["_summary"]["rate"],
            "document_leakage_post": results["document_leakage"]["post"]["_summary"]["rate"],
            "document_leakage_pre": results["document_leakage"]["pre"]["_summary"]["rate"],
            "retrieval_leakage_raw": results["retrieval_leakage"]["raw"]["_summary"]["rate"],
            "retrieval_leakage_post": results["retrieval_leakage"]["post"]["_summary"]["rate"],
            "retrieval_leakage_pre": results["retrieval_leakage"]["pre"]["_summary"]["rate"],
        }
        if has_llm:
            results["summary"]["phi_in_answer_raw"] = results["phi_in_answer"]["raw"]["_summary"]["rate"]
            results["summary"]["phi_in_answer_post"] = results["phi_in_answer"]["post"]["_summary"]["rate"]
            results["summary"]["phi_in_answer_pre"] = results["phi_in_answer"]["pre"]["_summary"]["rate"]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n  Document Leakage:")
    print(f"    raw:   {results['document_leakage']['raw']['_summary']['rate']:.1%}")
    print(f"    post:  {results['document_leakage']['post']['_summary']['rate']:.1%}")
    print(f"    pre:   {results['document_leakage']['pre']['_summary']['rate']:.1%}")

    print("\n  Retrieval Leakage (k=5):")
    print(f"    raw:   {results['retrieval_leakage']['raw']['_summary']['rate']:.1%}")
    print(f"    post:  {results['retrieval_leakage']['post']['_summary']['rate']:.1%}")
    print(f"    pre:   {results['retrieval_leakage']['pre']['_summary']['rate']:.1%}")

    print("\n  Masking Recall:")
    for field, data in results["masking_recall"].items():
        if "recall" in data:
            print(f"    {field:<12}: {data['recall']:.1%}")

    if overall_recall:
        print(f"    {'OVERALL':<12}: {sum(overall_recall)/len(overall_recall):.1%}")

    if has_llm:
        print("\n  PHI in Answers:")
        print(f"    raw:   {results['phi_in_answer']['raw']['_summary']['rate']:.1%}")
        print(f"    post:  {results['phi_in_answer']['post']['_summary']['rate']:.1%}")
        print(f"    pre:   {results['phi_in_answer']['pre']['_summary']['rate']:.1%}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    run_evaluation()
