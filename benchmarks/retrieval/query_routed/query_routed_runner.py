#!/usr/bin/env python3
"""
Query-routed fixed-depth retrieval — research benchmark.

Compares:
  Fixed K=2
  Fixed K=20
  Routed: single->2, multi->20 (heuristic lexical classifier)

Uses:
  - ground_truth_v2.json (605 queries, includes 5 aggregates)
  - metrics.py canonical definitions (hit_rate, precision, recall, MRR)
  - dense retrieval infra (medical masking, embedding, FAISS)
  - query_router classification (lexical, runtime-only)

Does NOT:
  - modify ground truth
  - duplicate metric definitions
  - use MRN mappings in routing
  - change embedding / masking / FAISS / generation

Reports:
  - HitRate, Precision, Recall, MRR for overall / single-target / multi-record / each aggregate
  - routed K distribution (mean, median, min, max, counts for 2 vs 20)
"""

import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmarks.retrieval.ground_truth import GROUND_TRUTH_PATH
from benchmarks.retrieval.metrics import hit_rate_at_k, precision_at_k, recall_at_k, mrr_at_k
from secure_rag.detection import load_detector_stack
from secure_rag.embedding import embed_chunks
from secure_rag.masker import mask_text
from secure_rag.pdf_loader import chunk_text
from secure_rag.policies import load_policy
from secure_rag.query_router import AGGREGATE_K, DEFAULT_SINGLE_K, classify_query
from secure_rag.vector_store import VectorStore

RETRIEVAL_DIR = Path(__file__).parent
PROJECT_DIR = Path(__file__).parent.parent.parent.parent
RESULTS_PATH = RETRIEVAL_DIR / "query_routed_results.json"
GROUND_TRUTH_FILE = GROUND_TRUTH_PATH


def _load_mrn_records_raw():
    text = Path(PROJECT_DIR / "data" / "sample_patient_data.txt").read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]
    records = {}
    for block in blocks:
        m = re.search(r"Medical ID:\s*(MRN\d+)", block)
        if not m:
            continue
        rid = m.group(1)
        records[rid] = {"record_id": rid, "text": block}
    return records


def _build_masked_index(records):
    policy = load_policy("medical")
    detectors = load_detector_stack("medical")
    texts_with_ids = []
    for rid in sorted(records.keys()):
        r = records[rid]
        text = mask_text(r["text"], policy=policy, detectors=detectors)
        texts_with_ids.append((rid, text))
    chunks = []
    chunk_record_map = []
    for rid, text in texts_with_ids:
        cs = chunk_text(text)
        chunks.extend(cs)
        chunk_record_map.extend([rid] * len(cs))
    embeddings = embed_chunks(chunks)
    vector_store = VectorStore(embeddings)
    return vector_store, chunks, chunk_record_map


def _retrieve_top_k(query, vector_store, k):
    q_vec = embed_chunks([query])
    q_vec = np.array(q_vec).astype("float32")
    distances, indices = vector_store.search(q_vec, k=k)
    # Normalize distances/indices to lists
    if hasattr(distances, "tolist"):
        dl = distances.tolist()
        if isinstance(dl, list) and len(dl) > 0 and isinstance(dl[0], list):
            dl = dl[0]
    else:
        dl = list(distances)
    il = indices if isinstance(indices, list) else list(indices)
    if isinstance(il, list) and len(il) > 0 and isinstance(il[0], list):
        il = il[0]
    return il, dl


def _make_retrieved(indices, chunk_record_map, relevant_set):
    retrieved = []
    for rank, chunk_idx in enumerate(indices):
        rid = chunk_record_map[chunk_idx] if 0 <= chunk_idx < len(chunk_record_map) else "UNKNOWN"
        retrieved.append({
            "record_id": rid,
            "rank": rank,
            "chunk_index": int(chunk_idx),
            "relevant": rid in relevant_set,
        })
    return retrieved


def run_benchmark():
    print("=" * 60)
    print("Query-Routed Retrieval — Research Benchmark")
    print("=" * 60)

    gt = json.loads(GROUND_TRUTH_FILE.read_text())
    queries = gt["queries"]
    print(f"Ground truth: {len(queries)} queries (v={gt.get('version')})")

    # Splits
    single_queries = [q for q in queries if len(q["relevant_records"]) == 1]
    multi_queries = [q for q in queries if len(q["relevant_records"]) > 1]
    agg_queries = [q for q in queries if q["qid"].startswith("AGG_")]
    print(f"  single-target (1 relevant): {len(single_queries)}")
    print(f"  multi-record (>1 relevant): {len(multi_queries)}")
    print(f"  aggregate queries (AGG_*): {len(agg_queries)}")

    records = _load_mrn_records_raw()
    print(f"\nLoaded {len(records)} records")
    assert len(records) == 120

    print("Building masked index (medical pre-embedding masking)...")
    vector_store, chunks, chunk_record_map = _build_masked_index(records)
    print(f"Index: {len(chunks)} chunks, ntotal={vector_store.index.ntotal}")

    # For efficiency, retrieve top 20 for each query once, then slice for K=2
    results = []
    routed_ks = []
    for entry in queries:
        qid = entry["qid"]
        question = entry["question"]
        relevant_set = set(entry["relevant_records"])
        # classify
        label = classify_query(question)
        routed = AGGREGATE_K if label == "multi" else DEFAULT_SINGLE_K
        routed_ks.append(routed)

        # retrieve top max(20, routed) =20
        max_k = max(AGGREGATE_K, DEFAULT_SINGLE_K)
        indices, distances = _retrieve_top_k(question, vector_store, k=max_k)
        retrieved_full = _make_retrieved(indices, chunk_record_map, relevant_set)

        # fixed K=2 slice
        ret_k2 = retrieved_full[:DEFAULT_SINGLE_K]
        ret_k20 = retrieved_full[:AGGREGATE_K]
        ret_routed = retrieved_full[:routed]

        # compute metrics
        metrics_k2 = {
            "hit_rate": hit_rate_at_k(ret_k2, DEFAULT_SINGLE_K, relevant_set),
            "precision": precision_at_k(ret_k2, DEFAULT_SINGLE_K, relevant_set),
            "recall": recall_at_k(ret_k2, DEFAULT_SINGLE_K, relevant_set=relevant_set),
            "mrr": mrr_at_k(ret_k2, DEFAULT_SINGLE_K, relevant_set),
        }
        metrics_k20 = {
            "hit_rate": hit_rate_at_k(ret_k20, AGGREGATE_K, relevant_set),
            "precision": precision_at_k(ret_k20, AGGREGATE_K, relevant_set),
            "recall": recall_at_k(ret_k20, AGGREGATE_K, relevant_set=relevant_set),
            "mrr": mrr_at_k(ret_k20, AGGREGATE_K, relevant_set),
        }
        metrics_routed = {
            "hit_rate": hit_rate_at_k(ret_routed, routed, relevant_set),
            "precision": precision_at_k(ret_routed, routed, relevant_set),
            "recall": recall_at_k(ret_routed, routed, relevant_set=relevant_set),
            "mrr": mrr_at_k(ret_routed, routed, relevant_set),
        }

        results.append({
            "qid": qid,
            "question": question,
            "relevant_records": sorted(relevant_set),
            "num_relevant": len(relevant_set),
            "expected_behaviour": entry.get("expected_behaviour"),
            "label": label,
            "routed_k": routed,
            "retrieved_record_ids_top20": [r["record_id"] for r in retrieved_full],
            "metrics_k2": metrics_k2,
            "metrics_k20": metrics_k20,
            "metrics_routed": metrics_routed,
        })

    # Aggregate helpers
    def aggregate(entries, metrics_key):
        if not entries:
            return {m: 0.0 for m in ["hit_rate", "precision", "recall", "mrr"]}
        out = {}
        for m in ["hit_rate", "precision", "recall", "mrr"]:
            vals = [e[metrics_key][m] for e in entries]
            out[m] = round(sum(vals) / len(vals), 6)
        return out

    overall = {
        "k2": aggregate(results, "metrics_k2"),
        "k20": aggregate(results, "metrics_k20"),
        "routed": aggregate(results, "metrics_routed"),
        "count": len(results),
    }
    single = {
        "k2": aggregate([r for r in results if r["num_relevant"] == 1], "metrics_k2"),
        "k20": aggregate([r for r in results if r["num_relevant"] == 1], "metrics_k20"),
        "routed": aggregate([r for r in results if r["num_relevant"] == 1], "metrics_routed"),
        "count": len(single_queries),
    }
    multi = {
        "k2": aggregate([r for r in results if r["num_relevant"] > 1], "metrics_k2"),
        "k20": aggregate([r for r in results if r["num_relevant"] > 1], "metrics_k20"),
        "routed": aggregate([r for r in results if r["num_relevant"] > 1], "metrics_routed"),
        "count": len(multi_queries),
    }

    # per-aggregate breakdown
    per_agg = {}
    for entry in [q for q in queries if q["qid"].startswith("AGG_")]:
        qid = entry["qid"]
        r = next(x for x in results if x["qid"] == qid)
        per_agg[qid] = {
            "question": r["question"],
            "num_relevant": r["num_relevant"],
            "label": r["label"],
            "routed_k": r["routed_k"],
            "k2": r["metrics_k2"],
            "k20": r["metrics_k20"],
            "routed": r["metrics_routed"],
        }

    # routed distribution
    routed_dist = {
        "values": routed_ks,
        "mean": round(statistics.mean(routed_ks), 4) if routed_ks else 0,
        "median": round(statistics.median(routed_ks), 4) if routed_ks else 0,
        "min": min(routed_ks) if routed_ks else 0,
        "max": max(routed_ks) if routed_ks else 0,
        "count_k2": routed_ks.count(DEFAULT_SINGLE_K),
        "count_k20": routed_ks.count(AGGREGATE_K),
        "total": len(routed_ks),
    }

    output = {
        "version": "query_routed_v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "description": "Query-routed fixed-depth retrieval benchmark. Fixed K=2, Fixed K=20, Routed single->2 multi->20. Dense FAISS, medical masking, lexical routing.",
        "k_values": {"single": DEFAULT_SINGLE_K, "aggregate": AGGREGATE_K},
        "ground_truth_version": gt.get("version"),
        "total_records": len(records),
        "num_chunks": len(chunks),
        "total_queries": len(queries),
        "statistics": {
            "single_target_count": len(single_queries),
            "multi_record_count": len(multi_queries),
            "aggregate_query_count": len(agg_queries),
        },
        "metrics": {
            "overall": overall,
            "single_target": single,
            "multi_record": multi,
            "per_aggregate": per_agg,
        },
        "routed_distribution": routed_dist,
        "per_query": results,
    }

    return output


def save_results(results, path=None):
    if path is None:
        path = RESULTS_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def print_summary(results):
    print("\n" + "=" * 60)
    print("QUERY-ROUTED BENCHMARK — SUMMARY")
    print("=" * 60)
    m = results["metrics"]
    for group in ["overall", "single_target", "multi_record"]:
        print(f"\n[{group}] n={m[group]['count']}")
        for mode in ["k2", "k20", "routed"]:
            vals = m[group][mode]
            print(f"  {mode:6s} HitRate={vals['hit_rate']:.4f} Prec={vals['precision']:.4f} Recall={vals['recall']:.4f} MRR={vals['mrr']:.4f}")

    print("\n[per-aggregate queries]")
    for qid, data in results["metrics"]["per_aggregate"].items():
        print(f"  {qid} (n={data['num_relevant']} label={data['label']} K={data['routed_k']})")
        for mode in ["k2", "k20", "routed"]:
            v = data[mode]
            print(f"    {mode:6s} HR={v['hit_rate']:.3f} P={v['precision']:.3f} R={v['recall']:.3f} MRR={v['mrr']:.3f}")

    rd = results["routed_distribution"]
    print(f"\n[routed distribution] total={rd['total']}  K=2:{rd['count_k2']}  K=20:{rd['count_k20']}")
    print(f"  mean={rd['mean']} median={rd['median']} min={rd['min']} max={rd['max']}")
    print("=" * 60)


if __name__ == "__main__":
    res = run_benchmark()
    out = save_results(res)
    print(f"\nResults saved to {out}")
    print_summary(res)
