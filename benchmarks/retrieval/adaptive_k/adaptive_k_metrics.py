#!/usr/bin/env python3
"""
Adaptive K — Metrics (research-only)

Computes per spec:
  HitRate: 1 if at least one relevant retrieved else 0
  Precision: |relevant ∩ retrieved| / K
  Recall: |relevant ∩ retrieved| / |relevant|
  MRR: reciprocal rank of first relevant record

Reuses definitions from benchmarks/retrieval/metrics.py
Aggregates by K:
  mean HitRate, mean Precision, mean Recall, mean MRR
Also separate:
  A. all aggregate queries (5)
  B. multi-record queries (4 with >1 relevant)
  C. single-target aggregate query (T2D+Hypertension, 1 relevant)
Also computes per-query K thresholds for 80/90/100% recall.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from benchmarks.retrieval.metrics import hit_rate_at_k, mrr_at_k, precision_at_k, recall_at_k

RETRIEVAL_DIR = Path(__file__).parent
RESULTS_PATH = RETRIEVAL_DIR / "adaptive_k_retrieval_results.json"
METRICS_PATH = RETRIEVAL_DIR / "adaptive_k_metrics.json"

K_VALUES = [2, 5, 10, 20, 30, 50]

# Ground truth queries
MULTI_QIDS = ["AGG_HYPERTENSION", "AGG_AMLODIPINE_5MG", "AGG_PARACETAMOL_650MG", "AGG_METFORMIN_500MG"]
SINGLE_QIDS = ["AGG_T2D_HYPERTENSION"]


def _mk_retrieved_list(record_ids: List[str]):
    """Convert list of record_ids to retrieved format expected by metrics helpers."""
    return [{"record_id": rid, "relevant": False, "rank": i, "chunk_index": i, "score": 0.0} for i, rid in enumerate(record_ids)]


def compute_adaptive_k_metrics(retrieval_results: dict = None) -> dict:
    if retrieval_results is None:
        with open(RESULTS_PATH) as f:
            retrieval_results = json.load(f)

    k_values = retrieval_results.get("k_values", K_VALUES)
    results_by_qid = {r["qid"]: r for r in retrieval_results["results"]}

    per_query = {}
    for qid, entry in results_by_qid.items():
        gt_set = set(entry["relevant_records"])
        total_relevant = len(gt_set)
        q_metrics = {
            "qid": qid,
            "question": entry["question"],
            "total_relevant": total_relevant,
            "expected_behaviour": entry.get("expected_behaviour", ""),
        }
        for k in k_values:
            k_str = str(k)
            per_k_data = entry["per_k"][k_str]
            retrieved_ids = per_k_data["retrieved_record_ids"]
            retrieved = _mk_retrieved_list(retrieved_ids)
            hr = hit_rate_at_k(retrieved, k, gt_set)
            prec = precision_at_k(retrieved, k, gt_set)
            rec = recall_at_k(retrieved, k, relevant_set=gt_set)
            mrr = mrr_at_k(retrieved, k, gt_set)
            relevant_retrieved = per_k_data["num_relevant_retrieved"]
            q_metrics[f"k_{k}"] = {
                "hit_rate": hr,
                "precision": prec,
                "recall": rec,
                "mrr": mrr,
                "relevant_retrieved": relevant_retrieved,
                "total_relevant": total_relevant,
            }
        per_query[qid] = q_metrics

    # Aggregated by K
    aggregated = {}
    # Helper to compute mean
    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    # A. all aggregate queries (5)
    for k in k_values:
        k_key = f"k_{k}"
        hits = [per_query[qid][k_key]["hit_rate"] for qid in per_query]
        precs = [per_query[qid][k_key]["precision"] for qid in per_query]
        recs = [per_query[qid][k_key]["recall"] for qid in per_query]
        mrrs = [per_query[qid][k_key]["mrr"] for qid in per_query]
        aggregated[k_key] = {
            "all": {
                "mean_hit_rate": round(_mean(hits), 6),
                "mean_precision": round(_mean(precs), 6),
                "mean_recall": round(_mean(recs), 6),
                "mean_mrr": round(_mean(mrrs), 6),
                "count": len(per_query),
            }
        }

    # B. multi-record queries
    for k in k_values:
        k_key = f"k_{k}"
        hits = [per_query[qid][k_key]["hit_rate"] for qid in MULTI_QIDS if qid in per_query]
        precs = [per_query[qid][k_key]["precision"] for qid in MULTI_QIDS if qid in per_query]
        recs = [per_query[qid][k_key]["recall"] for qid in MULTI_QIDS if qid in per_query]
        mrrs = [per_query[qid][k_key]["mrr"] for qid in MULTI_QIDS if qid in per_query]
        aggregated[k_key]["multi_record"] = {
            "mean_hit_rate": round(_mean(hits), 6),
            "mean_precision": round(_mean(precs), 6),
            "mean_recall": round(_mean(recs), 6),
            "mean_mrr": round(_mean(mrrs), 6),
            "count": len(MULTI_QIDS),
        }

    # C. single-target aggregate
    for k in k_values:
        k_key = f"k_{k}"
        hits = [per_query[qid][k_key]["hit_rate"] for qid in SINGLE_QIDS if qid in per_query]
        precs = [per_query[qid][k_key]["precision"] for qid in SINGLE_QIDS if qid in per_query]
        recs = [per_query[qid][k_key]["recall"] for qid in SINGLE_QIDS if qid in per_query]
        mrrs = [per_query[qid][k_key]["mrr"] for qid in SINGLE_QIDS if qid in per_query]
        aggregated[k_key]["single_target"] = {
            "mean_hit_rate": round(_mean(hits), 6),
            "mean_precision": round(_mean(precs), 6),
            "mean_recall": round(_mean(recs), 6),
            "mean_mrr": round(_mean(mrrs), 6),
            "count": len(SINGLE_QIDS),
        }

    # Per-query thresholds: min K achieving 80/90/100% recall
    thresholds = {}
    for qid, qm in per_query.items():
        thr = {}
        for level in [0.8, 0.9, 1.0]:
            found = None
            for k in sorted(k_values):
                rec = qm[f"k_{k}"]["recall"]
                if rec >= level - 1e-9:
                    found = k
                    break
            thr[str(level)] = found
        thresholds[qid] = {
            "question": qm["question"],
            "total_relevant": qm["total_relevant"],
            "thresholds": thr,
            "per_k_recall": {f"k_{k}": qm[f"k_{k}"]["recall"] for k in k_values},
            "per_k_precision": {f"k_{k}": qm[f"k_{k}"]["precision"] for k in k_values},
        }

    metrics = {
        "version": "adaptive_k_v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "description": "Adaptive K metrics: HitRate, Precision, Recall, MRR per K for 5 aggregate queries. Reuses benchmark definitions. Aggregates: all, multi-record (4), single-target (1).",
        "k_values": k_values,
        "per_query": per_query,
        "aggregated": aggregated,
        "thresholds": thresholds,
        "metric_definitions": {
            "hit_rate": "1 if at least one relevant record in top-k else 0 (0 if no relevant records)",
            "precision": "Unique relevant retrieved / k (deduplicated)",
            "recall": "Unique relevant retrieved / total relevant (0 if no relevant records)",
            "mrr": "1/rank of first relevant record (0 if none)",
        },
        "source_artifact": "adaptive_k_retrieval_results.json",
    }
    return metrics


def save_metrics(metrics: dict, path=None) -> Path:
    if path is None:
        path = METRICS_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


def load_metrics(path=None) -> dict:
    if path is None:
        path = METRICS_PATH
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    print("Adaptive K — Metrics")
    m = compute_adaptive_k_metrics()
    p = save_metrics(m)
    print(f"Metrics saved to {p}")
    # Print summary
    print("\nAggregated by K (all 5 queries):")
    for k in m["k_values"]:
        agg = m["aggregated"][f"k_{k}"]["all"]
        print(f"  K={k:2d}  HitRate={agg['mean_hit_rate']:.3f}  Prec={agg['mean_precision']:.3f}  Recall={agg['mean_recall']:.3f}  MRR={agg['mean_mrr']:.3f}")
    print("\nMulti-record (4):")
    for k in m["k_values"]:
        agg = m["aggregated"][f"k_{k}"]["multi_record"]
        print(f"  K={k:2d}  HitRate={agg['mean_hit_rate']:.3f}  Prec={agg['mean_precision']:.3f}  Recall={agg['mean_recall']:.3f}")
    print("\nThresholds (min K for recall levels):")
    for qid, thr in m["thresholds"].items():
        print(f"  {qid} ({thr['total_relevant']} relevant): 80%={thr['thresholds']['0.8']} 90%={thr['thresholds']['0.9']} 100%={thr['thresholds']['1.0']}")
