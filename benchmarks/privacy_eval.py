#!/usr/bin/env python3
"""
Privacy evaluation for Secure RAG - 3-configuration comparison.

Compares three evaluation configurations:
- baseline_a:  Raw RAG — no masking anywhere
- baseline_b:  Post-retrieval privacy masking
- secure_rag:  Pre-embedding masking (Secure RAG)

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
from secure_rag.generator import generate_answer
from secure_rag.masker import mask_text
from secure_rag.pdf_loader import chunk_text
from secure_rag.embedding import embed_chunks
from secure_rag.retriever import retrieve
from secure_rag.vector_store import VectorStore

BENCHMARK_DIR = Path(__file__).parent
DATASET_PATH = BENCHMARK_DIR / "dataset.jsonl"
QUERIES_PATH = BENCHMARK_DIR / "dataset_queries.json"
SPLIT_PATH = BENCHMARK_DIR / "train_test_split.json"
RESULTS_PATH = BENCHMARK_DIR / "results.json"

PII_FIELDS = ["name", "aadhaar", "pan", "phone", "email", "mrn", "dob", "address"]
RETRIEVAL_K = 5

DEBUG = False
MAX_LLM_FAILURE_RATE = 0.30
LLM_RETRIES = 1

# ---------------------------------------------------------------------------
# Centralized evaluation configuration registry
# ---------------------------------------------------------------------------
# Each config defines:
#   id:          stable machine-readable identifier (used in JSON keys, dispatch)
#   display_name: human-readable name for console output
#   description:  research methodology summary
#   get_idx:      callable(raw_index, raw_chunks, pre_index, pre_chunks) -> (index, chunks)
#   answer:       callable(query, index, chunks) -> str
#
# Adding a new baseline requires one entry here — no other code changes.
# ---------------------------------------------------------------------------
EVALUATION_CONFIGS = [
    {
        "id": "baseline_a",
        "display_name": "Baseline A — Raw Retrieval-Augmented Generation",
        "description": "No masking anywhere — documents, chunks, embeddings, and retrieved context are all raw.",
        "get_idx": lambda r_idx, r_ck, p_idx, p_ck: (r_idx, r_ck),
        "answer": lambda q, idx, ck: benchmark_answer(q, idx, ck, "baseline_a"),
    },
    {
        "id": "baseline_b",
        "display_name": "Baseline B — Post-Retrieval Privacy Masking",
        "description": "Raw index and retrieval; mask_text() applied to retrieved context before LLM generation.",
        "get_idx": lambda r_idx, r_ck, p_idx, p_ck: (r_idx, r_ck),
        "answer": lambda q, idx, ck: benchmark_answer(q, idx, ck, "baseline_b"),
    },
    {
        "id": "secure_rag",
        "display_name": "Proposed Method — Secure RAG",
        "description": "Full Secure RAG pipeline: pre-embedding masking, masked index, canonical runtime for answering.",
        "get_idx": lambda r_idx, r_ck, p_idx, p_ck: (p_idx, p_ck),
        "answer": lambda q, idx, ck: benchmark_answer(q, idx, ck, "secure_rag"),
    },
]


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


def truncate_at_stop_marker(text: str) -> str:
    stop_markers = ("\nContext:", "\nQuestion:", "[/INST]")
    positions = [text.find(m) for m in stop_markers if m in text]
    if positions:
        text = text[: min(positions)]
    return text.strip()


def benchmark_answer(query: str, index, chunks: List[str], config_id: str) -> str:
    if config_id == "secure_rag":
        return "".join(list(rag_answer(query, index, chunks))).strip()

    context_chunks = retrieve(query, index, chunks)
    context = "\n\n".join(chunk for chunk in context_chunks if chunk)

    if config_id == "baseline_b":
        context = mask_text(context)
    elif config_id != "baseline_a":
        raise ValueError(f"Unsupported benchmark config: {config_id}")

    response = "".join(generate_answer(context, f"{query}\n\nAnswer:"))
    return truncate_at_stop_marker(response)

def generate_answer_with_retry(query: str, index, chunks: List[str], config_id: str) -> str:
    last_error = None
    for attempt in range(LLM_RETRIES + 1):
        try:
            answer = benchmark_answer(query, index, chunks, config_id)
            if not answer:
                raise RuntimeError("LLM returned an empty answer")
            return answer
        except Exception as exc:
            last_error = exc
            if attempt == LLM_RETRIES:
                raise RuntimeError(
                    f"LLM evaluation failed for config={config_id}, query={query!r} after {LLM_RETRIES + 1} attempts"
                ) from exc
    raise RuntimeError("Unreachable LLM retry state") from last_error


def run_evaluation():
    print("=" * 60)
    print("Secure RAG Privacy Evaluation - Benchmark Comparison")
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

    for config in EVALUATION_CONFIGS:
        _, chunks = config["get_idx"](raw_index, raw_chunks, pre_index, pre_chunks)
        results["document_leakage"][config["id"]] = {}

        total_leaked = 0
        total_pii = 0

        for rid, record in test_records.items():
            leaked, total = compute_doc_leakage(chunks, record["pii"])
            total_leaked += leaked
            total_pii += total

            results["document_leakage"][config["id"]][rid] = {
                "leaked": leaked,
                "total": total,
            }

        results["document_leakage"][config["id"]]["_summary"] = {
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

    for config in EVALUATION_CONFIGS:
        index, chunks = config["get_idx"](raw_index, raw_chunks, pre_index, pre_chunks)
        results["retrieval_leakage"][config["id"]] = {}

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
            valid = 0
            failed = 0
            for field in PII_FIELDS:
                value = record["pii"].get(field)
                if not value:
                    continue
                total += 1
                if normalize(value) in normalize(all_text) or loose_match(value, all_text):
                    leaked += 1

            total_leaked += leaked
            total_pii += total

            results["retrieval_leakage"][config["id"]][rid] = {
                "leaked": leaked,
                "total": total,
            }

        results["retrieval_leakage"][config["id"]]["_summary"] = {
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

        for config in EVALUATION_CONFIGS:
            index, chunks = config["get_idx"](raw_index, raw_chunks, pre_index, pre_chunks)
            results["phi_in_answer"][config["id"]] = {}

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
                        answer = generate_answer_with_retry(
                            q["question"],
                            index,
                            chunks,
                            config_id=config["id"],
                        )
                        has_phi = check_phi_in_text(answer, record)
                        leaked += 1 if has_phi else 0
                        valid += 1
                    except Exception as exc:
                        failed += 1
                        print(f"  {config['display_name']}: FAILED {q['qid']} - {exc}")
                    total += 1

                if (i + 1) % 5 == 0:
                    print(f"  {config['display_name']}: Processed {i + 1}/{min(20, len(test_ids))}...")

            failure_rate = failed / total if total > 0 else 0.0
            if failure_rate > MAX_LLM_FAILURE_RATE:
                raise RuntimeError(
                    f"LLM failure rate too high for config={config['id']}: {failed}/{total} ({failure_rate:.1%})"
                )

            results["phi_in_answer"][config["id"]]["_summary"] = {
                "leaked": leaked,
                "total": total,
                "valid": valid,
                "failed": failed,
                "rate": leaked / valid if valid > 0 else 0.0,
            }
    else:
        print("\n" + "-" * 40)
        print("LLM not available (HF_API_KEY not set)")
        print("Skipping PHI-in-answer evaluation")
        print("-" * 40)

    overall_recall = [v["recall"] for v in results["masking_recall"].values() if "recall" in v]
    if overall_recall:
        results["summary"] = {"masking_recall": sum(overall_recall) / len(overall_recall)}
        for config in EVALUATION_CONFIGS:
            k = config["id"]
            results["summary"][f"document_leakage_{k}"] = results["document_leakage"][k]["_summary"]["rate"]
            results["summary"][f"retrieval_leakage_{k}"] = results["retrieval_leakage"][k]["_summary"]["rate"]
        if has_llm:
            for config in EVALUATION_CONFIGS:
                results["summary"][f"phi_in_answer_{config['id']}"] = results["phi_in_answer"][config["id"]]["_summary"]["rate"]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n  Document Leakage:")
    for config in EVALUATION_CONFIGS:
        print(f"    {config['id']}:   {results['document_leakage'][config['id']]['_summary']['rate']:.1%}")

    print("\n  Retrieval Leakage (k=5):")
    for config in EVALUATION_CONFIGS:
        print(f"    {config['id']}:   {results['retrieval_leakage'][config['id']]['_summary']['rate']:.1%}")

    print("\n  Masking Recall:")
    for field, data in results["masking_recall"].items():
        if "recall" in data:
            print(f"    {field:<12}: {data['recall']:.1%}")

    if overall_recall:
        print(f"    {'OVERALL':<12}: {sum(overall_recall)/len(overall_recall):.1%}")

    if has_llm:
        print("\n  PHI in Answers:")
        for config in EVALUATION_CONFIGS:
            summary = results['phi_in_answer'][config['id']]['_summary']
            print(
                f"    {config['id']}:   {summary['rate']:.1%} "
                f"(leaked={summary['leaked']}, valid={summary['valid']}, failed={summary['failed']}, total={summary['total']})"
            )

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    run_evaluation()
