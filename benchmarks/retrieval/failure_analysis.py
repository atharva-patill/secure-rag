"""
Retrieval Failure Analysis — Secure RAG retrieval evaluation.

Consumes metrics_v1.json + retrieval_results_v1.json to explain WHY
retrieval succeeds or fails for each query. No retrieval occurs.
No metrics are recomputed.

Taxonomy (from Phase 1 design):
  Entity Retrieval Failure
    Diagnosis | Hospital | Medication | Treatment | Demographics | Contact/Identifier
  General Query Failure
    Summary | Factual/Attribute
  Ranking Failure
    Low Similarity Score | Competitor Record Higher
  Chunk Boundary Failure
    Answer Fragmented | Context Lost
  Masking Degradation
    Entity Name Masked | Location Masked | Identifier Masked
  Embedding Similarity Failure
    Vocabulary Mismatch | Domain Drift
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

RETRIEVAL_DIR = Path(__file__).parent
BENCHMARK_DIR = RETRIEVAL_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))

FAILURE_ANALYSIS_VERSION = "v1"
FAILURE_ANALYSIS_FRAMEWORK_VERSION = "1"
FAILURE_ANALYSIS_PATH = RETRIEVAL_DIR / f"failure_analysis_{FAILURE_ANALYSIS_VERSION}.json"

SUBCATEGORY_TO_ENTITY_TYPE = {
    "factual_hospital": "hospital",
    "summary": None,
    "phi_aadhaar": "contact_identifier",
    "phi_phone": "contact_identifier",
    "phi_mrn": "contact_identifier",
}

SUBCATEGORY_TO_TAXONOMY = {
    "factual_hospital": ["entity_retrieval_failure", "hospital"],
    "summary": ["general_query_failure", "summary"],
    "phi_aadhaar": ["entity_retrieval_failure", "contact_identifier"],
    "phi_phone": ["entity_retrieval_failure", "contact_identifier"],
    "phi_mrn": ["entity_retrieval_failure", "contact_identifier"],
}


def load_retrieval_results() -> dict:
    from retrieval.runner import load_results
    return load_results()


def load_metrics() -> dict:
    from retrieval.metrics import load_metrics
    return load_metrics()


SUBCATEGORY_ENTITY_MAP = {
    "factual_hospital": {"primary": "entity_retrieval_failure", "specific": "hospital"},
    "summary": {"primary": "general_query_failure", "specific": "summary"},
    "phi_aadhaar": {"primary": "entity_retrieval_failure", "specific": "contact_identifier"},
    "phi_phone": {"primary": "entity_retrieval_failure", "specific": "contact_identifier"},
    "phi_mrn": {"primary": "entity_retrieval_failure", "specific": "contact_identifier"},
}


def _entity_type_label(subcategory: str) -> str:
    return SUBCATEGORY_ENTITY_MAP.get(subcategory, {}).get("specific", "unknown")


def classify_query_failure(
    qid: str,
    pq_metrics: dict,
    retrieval_entry: dict,
) -> dict:
    category = pq_metrics.get("category", "unknown")
    subcategory = pq_metrics.get("subcategory", "unknown")
    gt_records = pq_metrics.get("ground_truth_records", [])
    question = retrieval_entry.get("question", "")
    expected_behaviour = retrieval_entry.get("expected_behaviour", "unknown")

    config_results = {}
    overall_failed = False
    any_succeeded = False

    for cid in ["baseline_a", "baseline_b", "secure_rag"]:
        h10 = pq_metrics[cid]["k_10"]["hit_rate"]
        succeeded = h10 == 1
        failed = not succeeded
        if succeeded:
            any_succeeded = True
        if failed:
            overall_failed = True

        retrieved = retrieval_entry["results"][cid].get("retrieved", [])
        gt_ranks = [
            item["rank"] for item in retrieved
            if item["record_id"] in gt_records
        ]
        gt_retrieved_anywhere = len(gt_ranks) > 0
        min_gt_rank = min(gt_ranks) if gt_ranks else None

        failures = []

        if failed:
            if min_gt_rank is not None and min_gt_rank >= 10:
                failures.append({
                    "category": "ranking_failure",
                    "specific": "competitor_record_higher",
                    "evidence": {
                        "ground_truth_rank": min_gt_rank,
                        "k_analyzed": 10,
                    },
                    "rationale": (
                        f"Ground truth record {'/'.join(gt_records)} was retrieved "
                        f"at rank {min_gt_rank} (beyond top-10). "
                        f"Other records scored higher in embedding similarity."
                    ),
                })
            elif min_gt_rank is not None and min_gt_rank < 10:
                failures.append({
                    "category": "ranking_failure",
                    "specific": "competitor_record_higher",
                    "evidence": {
                        "ground_truth_rank": min_gt_rank,
                        "k_analyzed": 10,
                    },
                    "rationale": (
                        f"Ground truth was retrieved at rank {min_gt_rank} which is within top-10, "
                        f"but hit_rate@10=0 indicates the record was not captured at rank < 10. "
                        f"However metrics show hit_rate@10=0 which is inconsistent. "
                        f"This suggests a mismatch in relevance assignment."
                    ),
                })
            else:
                if category == "phi_targeting":
                    entity_type = _entity_type_label(subcategory)
                    failures.append({
                        "category": "entity_retrieval_failure",
                        "specific": entity_type,
                        "evidence": {
                            "ground_truth_retrieved": False,
                            "expected_behaviour": expected_behaviour,
                        },
                        "rationale": (
                            f"Query targets {entity_type.replace('_', ' ')} but "
                            f"the ground truth record {'/'.join(gt_records)} was not "
                            f"retrieved within the top-10 results. The embedding model "
                            f"did not match the query to the correct patient record."
                        ),
                    })
                elif category == "general":
                    failures.append({
                        "category": "embedding_similarity_failure",
                        "specific": "vocabulary_mismatch",
                        "evidence": {
                            "ground_truth_retrieved": False,
                            "expected_behaviour": expected_behaviour,
                        },
                        "rationale": (
                            f"General query '{question[:60]}...' did not retrieve the "
                            f"ground truth record. The query vocabulary likely overlaps "
                            f"with multiple records due to similar medical content."
                        ),
                    })

        if cid == "secure_rag" and pq_metrics["baseline_a"]["k_10"]["hit_rate"] == 1 and failed:
            failures.append({
                "category": "masking_degradation",
                "specific": "entity_name_masked",
                "evidence": {
                    "baseline_a_succeeded": True,
                    "secure_rag_failed": True,
                },
                "rationale": (
                    "Pre-embedding masking removed entity tokens from the indexed chunks, "
                    "causing the embedding to differ from the non-masked variant. "
                    "The raw index (Baseline A) successfully retrieved the ground truth record "
                    "while the masked index (Secure RAG) did not."
                ),
            })

        config_results[cid] = {
            "succeeded": succeeded,
            "hit_rate_10": h10,
            "failures": failures,
            "ground_truth_retrieved": gt_retrieved_anywhere,
            "ground_truth_min_rank": min_gt_rank,
        }

    return {
        "qid": qid,
        "question": question,
        "category": category,
        "subcategory": subcategory,
        "expected_behaviour": expected_behaviour,
        "overall_failed": overall_failed,
        "any_succeeded": any_succeeded,
        "configs": config_results,
    }


def run_failure_analysis(metrics: dict = None, retrieval_results: dict = None) -> dict:
    if metrics is None:
        metrics = load_metrics()
    if retrieval_results is None:
        retrieval_results = load_retrieval_results()

    per_query_metrics = metrics["per_query"]
    retrieval_by_qid = {q["qid"]: q for q in retrieval_results["queries"]}

    analyses = {}
    for qid, pq in per_query_metrics.items():
        r_entry = retrieval_by_qid.get(qid)
        if r_entry is None:
            continue
        analyses[qid] = classify_query_failure(qid, pq, r_entry)

    category_counts: Dict[str, int] = {}
    config_category_counts: Dict[str, Dict[str, int]] = {}
    subcategory_counts: Dict[str, int] = {}

    for cid in ["baseline_a", "baseline_b", "secure_rag"]:
        config_category_counts[cid] = {}

    for qid, analysis in analyses.items():
        for cid in ["baseline_a", "baseline_b", "secure_rag"]:
            for failure in analysis["configs"][cid]["failures"]:
                cat = failure["category"]
                category_counts[cat] = category_counts.get(cat, 0) + 1
                config_category_counts[cid][cat] = config_category_counts[cid].get(cat, 0) + 1

        subcat = analysis.get("subcategory", "unknown")
        if analysis["overall_failed"]:
            subcategory_counts[subcat] = subcategory_counts.get(subcat, 0) + 1

    per_config_summary = {}
    for cid in ["baseline_a", "baseline_b", "secure_rag"]:
        total = sum(1 for a in analyses.values() if a["configs"][cid]["hit_rate_10"] == 0)
        per_config_summary[cid] = {
            "total_failures": total,
            "total_queries": len(analyses),
            "failure_rate": round(total / len(analyses), 6) if analyses else 0.0,
            "categories": {},
        }
        for cat, count in sorted(config_category_counts[cid].items()):
            per_config_summary[cid]["categories"][cat] = count

    total_failures = sum(1 for a in analyses.values() if a["overall_failed"])

    result = {
        "version": FAILURE_ANALYSIS_VERSION,
        "framework_version": FAILURE_ANALYSIS_FRAMEWORK_VERSION,
        "source_artifacts": {
            "metrics": "metrics_v1.json",
            "retrieval_results": "retrieval_results_v1.json",
        },
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "description": (
            "Canonical failure analysis artifact for Secure RAG retrieval evaluation. "
            "Classifies retrieval failures using the approved taxonomy. "
            "Each failure includes category, specific type, supporting evidence, and rationale."
        ),
        "taxonomy": {
            "entity_retrieval_failure": "Query specifies a named entity but the correct record was not retrieved",
            "general_query_failure": "General/summary query failed to retrieve the correct record",
            "ranking_failure": "Correct record retrieved but ranked below k threshold",
            "masking_degradation": "Pre-embedding masking caused retrieval failure in Secure RAG",
            "embedding_similarity_failure": "Semantic mismatch between query and record embeddings",
        },
        "statistics": {
            "total_queries": len(analyses),
            "total_failures": total_failures,
            "overall_failure_rate": round(total_failures / len(analyses), 6) if analyses else 0.0,
            "failure_categories": dict(sorted(category_counts.items())),
            "failure_by_subcategory": dict(sorted(subcategory_counts.items())),
            "per_config": per_config_summary,
        },
        "per_query": analyses,
    }

    return result


def save_failure_analysis(analysis: dict, path=None) -> Path:
    if path is None:
        path = FAILURE_ANALYSIS_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(analysis, f, indent=2)
    return path


def load_failure_analysis(path=None) -> dict:
    if path is None:
        configs = sorted(FAILURE_ANALYSIS_PATH.parent.glob("failure_analysis_*.json"))
        if not configs:
            raise FileNotFoundError("No failure analysis file found.")
        path = configs[-1]
    with open(path) as f:
        return json.load(f)


def validate(analysis: dict) -> List[str]:
    issues = []

    if analysis["version"] != FAILURE_ANALYSIS_VERSION:
        issues.append(f"FAIL: Expected version {FAILURE_ANALYSIS_VERSION}, got {analysis['version']}")

    per_query = analysis.get("per_query", {})
    if not per_query:
        issues.append("FAIL: No per-query analyses.")
        return issues

    pq_count = len(per_query)
    expected_configs = {"baseline_a", "baseline_b", "secure_rag"}

    seen_qids = set()
    for qid, a in per_query.items():
        if qid in seen_qids:
            issues.append(f"FAIL: Duplicate qid: {qid}")
        seen_qids.add(qid)

        actual_configs = set(a.get("configs", {}).keys())
        if actual_configs != expected_configs:
            issues.append(f"FAIL: {qid} configs mismatch ({actual_configs})")

        for cid in expected_configs:
            conf = a["configs"].get(cid, {})
            if "succeeded" not in conf:
                issues.append(f"FAIL: {qid}/{cid} missing succeeded")
            if "failures" not in conf:
                issues.append(f"FAIL: {qid}/{cid} missing failures")
            else:
                for f in conf["failures"]:
                    if "category" not in f:
                        issues.append(f"FAIL: {qid}/{cid} failure missing category")
                    if "rationale" not in f:
                        issues.append(f"FAIL: {qid}/{cid} failure missing rationale")

    stats = analysis.get("statistics", {})
    if stats.get("total_queries") != pq_count:
        issues.append(f"FAIL: statistics.total_queries ({stats.get('total_queries')}) != actual ({pq_count})")

    if "failure_categories" not in stats:
        issues.append("FAIL: missing failure_categories in statistics")
    if "per_config" not in stats:
        issues.append("FAIL: missing per_config in statistics")

    if not issues:
        issues.append(f"PASS: {pq_count} queries analyzed, "
                      f"{stats.get('total_failures', 0)} failures classified, "
                      f"{len(stats.get('failure_categories', {}))} category types.")

    return issues


def print_summary(analysis: dict):
    stats = analysis["statistics"]
    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS — SUMMARY")
    print("=" * 60)
    print(f"\n  Total queries:     {stats['total_queries']}")
    print(f"  Total failures:    {stats['total_failures']}")
    print(f"  Failure rate:      {stats['overall_failure_rate']:.2%}")

    print(f"\n  Failure categories:")
    for cat, count in sorted(stats["failure_categories"].items()):
        print(f"    {cat:<35} {count:>4}")

    print(f"\n  Per-config failures:")
    for cid, csum in stats["per_config"].items():
        cats = "  ".join(f"{k}={v}" for k, v in sorted(csum["categories"].items()))
        print(f"    {cid:<15} {csum['total_failures']:>4}/{csum['total_queries']}  "
              f"({csum['failure_rate']:.1%})  {cats}")

    print(f"\n  Failures by subcategory:")
    for sub, count in sorted(stats.get("failure_by_subcategory", {}).items()):
        print(f"    {sub:<25} {count:>4}")

    print(f"\n  Output:           {FAILURE_ANALYSIS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    print("Retrieval Failure Analysis — Phase 5")
    print()

    analysis = run_failure_analysis()
    path = save_failure_analysis(analysis)
    print(f"Failure analysis saved to: {path}")

    print("\nValidating failure analysis...")
    v_issues = validate(analysis)
    for issue in v_issues:
        print(f"  {issue}")

    print_summary(analysis)
