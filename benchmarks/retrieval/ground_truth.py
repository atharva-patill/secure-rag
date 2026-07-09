"""
Ground Truth Framework for Secure RAG retrieval evaluation.

Defines, validates, and exposes the canonical relevance judgments
for all 600 benchmark queries. Each query maps to exactly one
relevant record (its source record) in the current dataset.

Design:
- Record-centric: relevance is at the record level, not chunk level
- Each query has exactly one relevant record (its parent)
- Binary relevance: relevant or not-relevant
- Categories derived from query type (general vs PHI-targeting)
- Expected retrieval behaviour records the anticipated challenge level

Ground truth is versioned (v1) to allow future evolution.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

RETRIEVAL_DIR = Path(__file__).parent
BENCHMARK_DIR = RETRIEVAL_DIR.parent

sys.path.insert(0, str(BENCHMARK_DIR.parent))

GROUND_TRUTH_VERSION = "v1"
GROUND_TRUTH_PATH = RETRIEVAL_DIR / f"ground_truth_{GROUND_TRUTH_VERSION}.json"
DEFAULT_GROUND_TRUTH_PATH = GROUND_TRUTH_PATH

DATASET_PATH = BENCHMARK_DIR / "dataset.jsonl"
QUERIES_PATH = BENCHMARK_DIR / "dataset_queries.json"

QID_TO_SUBCATEGORY = {
    1: "factual_hospital",
    2: "summary",
    3: "phi_aadhaar",
    4: "phi_phone",
    5: "phi_mrn",
}

QID_TO_EXPECTED_BEHAVIOUR = {
    1: "record_retrieval",
    2: "record_retrieval",
    3: "entity_retrieval",
    4: "entity_retrieval",
    5: "entity_retrieval",
}


def _extract_query_number(qid: str) -> int:
    return int(qid.split("_Q")[-1])


def generate_ground_truth() -> dict:
    records = _load_records_raw()
    queries = _load_queries_raw()

    record_ids = set(records.keys())

    entries = []
    for group in queries:
        rid = group["record_id"]
        for q in group["queries"]:
            qid = q["qid"]
            qnum = _extract_query_number(qid)
            cat = "phi_targeting" if q.get("phi_in_answer") else "general"

            entries.append({
                "qid": qid,
                "question": q["question"],
                "category": cat,
                "subcategory": QID_TO_SUBCATEGORY.get(qnum, "unknown"),
                "relevant_records": [rid],
                "phi_in_answer": q.get("phi_in_answer", False),
                "expected_behaviour": QID_TO_EXPECTED_BEHAVIOUR.get(qnum, "unknown"),
            })

    ground_truth = {
        "version": GROUND_TRUTH_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "description": (
            "Ground truth for Secure RAG retrieval evaluation. "
            "Each query maps to its source record as the single relevant record. "
            "Binary relevance: the source record is relevant; all others are not."
        ),
        "schema": {
            "qid": "str - query identifier (e.g., MED117_Q1)",
            "question": "str - the query text",
            "category": "str - query category (general | phi_targeting)",
            "subcategory": "str - fine-grained category",
            "relevant_records": "list[str] - relevant record IDs",
            "phi_in_answer": "bool - whether the query targets PII",
            "expected_behaviour": "str - expected retrieval behaviour",
        },
        "statistics": {},
        "queries": entries,
    }

    stats = compute_statistics(ground_truth, records, record_ids)
    ground_truth["statistics"] = stats

    return ground_truth


def _load_records_raw() -> Dict[str, dict]:
    records = {}
    with open(DATASET_PATH) as f:
        for line in f:
            r = json.loads(line)
            records[r["record_id"]] = r
    return records


def _load_queries_raw() -> list:
    with open(QUERIES_PATH) as f:
        return json.load(f)


def compute_statistics(ground_truth: dict, records: dict = None, record_ids: set = None) -> dict:
    queries = ground_truth["queries"]

    if records is None:
        records = _load_records_raw()
        record_ids = set(records.keys())

    cat_counts: Dict[str, int] = {}
    subcat_counts: Dict[str, int] = {}
    behaviour_counts: Dict[str, int] = {}
    phi_count = 0
    qids_with_gt = 0

    for entry in queries:
        qids_with_gt += 1
        cat = entry["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

        subcat = entry["subcategory"]
        subcat_counts[subcat] = subcat_counts.get(subcat, 0) + 1

        behaviour = entry["expected_behaviour"]
        behaviour_counts[behaviour] = behaviour_counts.get(behaviour, 0) + 1

        if entry.get("phi_in_answer"):
            phi_count += 1

    all_records_exist = all(
        rid in record_ids
        for entry in queries
        for rid in entry["relevant_records"]
    )

    referenced_record_ids = set()
    for entry in queries:
        for rid in entry["relevant_records"]:
            referenced_record_ids.add(rid)

    return {
        "version": ground_truth["version"],
        "total_queries": len(queries),
        "total_records": len(record_ids),
        "queries_with_ground_truth": qids_with_gt,
        "phi_targeting_queries": phi_count,
        "general_queries": len(queries) - phi_count,
        "category_distribution": cat_counts,
        "subcategory_distribution": subcat_counts,
        "expected_behaviour_distribution": behaviour_counts,
        "all_records_exist": all_records_exist,
        "referenced_record_count": len(referenced_record_ids),
        "all_categories_valid": all(
            entry["category"] in ("general", "phi_targeting")
            for entry in queries
        ),
        "all_behaviours_valid": all(
            entry["expected_behaviour"] in ("record_retrieval", "entity_retrieval")
            for entry in queries
        ),
    }


def validate_ground_truth(ground_truth: dict = None) -> List[str]:
    if ground_truth is None:
        configs = list(GROUND_TRUTH_PATH.parent.glob(f"ground_truth_*.json"))
        if not configs:
            return ["No ground truth file found."]
        ground_truth = json.loads(Path(configs[-1]).read_text())

    issues = []
    records = _load_records_raw()
    record_ids = set(records.keys())
    queries = ground_truth.get("queries", [])

    if not queries:
        issues.append("FAIL: No queries in ground truth.")

    known_categories = {"general", "phi_targeting"}
    known_behaviours = {"record_retrieval", "entity_retrieval"}
    seen_qids = set()

    for entry in queries:
        qid = entry.get("qid")
        if not qid:
            issues.append("FAIL: Query missing qid.")
            continue

        if qid in seen_qids:
            issues.append(f"FAIL: Duplicate qid: {qid}")
        seen_qids.add(qid)

        cat = entry.get("category")
        if cat not in known_categories:
            issues.append(f"FAIL: {qid} has invalid category: {cat}")

        subcat = entry.get("subcategory")
        if subcat not in QID_TO_SUBCATEGORY.values() and subcat != "unknown":
            issues.append(f"FAIL: {qid} has invalid subcategory: {subcat}")

        rel = entry.get("relevant_records", [])
        if not rel:
            issues.append(f"FAIL: {qid} has no relevant records.")
        for rid in rel:
            if rid not in record_ids:
                issues.append(f"FAIL: {qid} references non-existent record: {rid}")

        behaviour = entry.get("expected_behaviour")
        if behaviour not in known_behaviours:
            issues.append(f"FAIL: {qid} has invalid expected_behaviour: {behaviour}")

    if not issues:
        stats = ground_truth.get("statistics", {})
        issues.append(f"PASS: {stats.get('total_queries', len(queries))} queries validated.")
        issues.append(f"PASS: {stats.get('total_records', len(record_ids))} records available.")
        issues.append(f"PASS: {stats.get('referenced_record_count', 0)} records referenced.")
        if stats.get("all_records_exist"):
            issues.append("PASS: All referenced records exist.")
        if stats.get("all_categories_valid"):
            issues.append("PASS: All categories are valid.")
        if stats.get("all_behaviours_valid"):
            issues.append("PASS: All expected_behaviour values are valid.")

    return issues


def save_ground_truth(ground_truth: dict, path=None) -> Path:
    if path is None:
        path = GROUND_TRUTH_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    return path


def load_ground_truth(path=None) -> dict:
    if path is None:
        configs = sorted(GROUND_TRUTH_PATH.parent.glob("ground_truth_*.json"))
        if not configs:
            raise FileNotFoundError("No ground truth file found in benchmarks/retrieval/")
        path = configs[-1]
    with open(path) as f:
        return json.load(f)


def format_validation(issues: List[str]) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("Ground Truth Validation")
    lines.append("=" * 60)
    for issue in issues:
        lines.append(f"  {issue}")
    lines.append("-" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    print("Generating ground truth...")
    gt = generate_ground_truth()

    path = save_ground_truth(gt)
    print(f"Saved to: {path}")

    print("\nValidation:")
    issues = validate_ground_truth(gt)
    print(format_validation(issues))

    stats = gt["statistics"]
    print(f"\nStatistics:")
    print(f"  Version:            {stats['version']}")
    print(f"  Total queries:      {stats['total_queries']}")
    print(f"  Total records:      {stats['total_records']}")
    print(f"  PHI-targeting:      {stats['phi_targeting_queries']}")
    print(f"  General:            {stats['general_queries']}")
    print(f"  Referenced records: {stats['referenced_record_count']}")

    for cat, count in stats["category_distribution"].items():
        print(f"  Category '{cat}':   {count}")
    for subcat, count in stats["subcategory_distribution"].items():
        print(f"    - {subcat}:        {count}")
    for beh, count in stats["expected_behaviour_distribution"].items():
        print(f"  Behaviour '{beh}': {count}")
