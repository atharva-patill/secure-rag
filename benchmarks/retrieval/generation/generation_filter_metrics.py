"""
generation_filter_metrics — research-only helper for filtering experiment.

Reuses v2 masked metrics and provides comparison against v2 baseline at
same budgets (800, 1200). Does NOT invent new scoring; re-exports.

Also provides delta helpers for filtered minus baseline.
"""

import json
from pathlib import Path
from typing import Dict

from .generation_metrics import (
    EXPECTED_HYPERTENSION_COUNT,
    EXPECTED_K,
    EXPECTED_HYPERTENSION_MRNS,
    aggregate_by_budget,
    compute_masked_generation_metrics,
    build_masked_record_fingerprints,
)

FILTER_INSTRUCTION = (
    "Enumerate only records whose Diagnosis contains Hypertension. "
    "Do not enumerate retrieved records whose Diagnosis does not contain Hypertension. "
    "Do not infer, combine, or invent patient records. "
    "Each enumerated patient must correspond to a retrieved record whose Diagnosis explicitly contains Hypertension."
)

GENERATION_DIR = Path(__file__).parent
V2_METRICS_PATH = GENERATION_DIR / "generation_metrics_v2.json"
FILTERED_METRICS_PATH = GENERATION_DIR / "generation_filter_metrics.json"
V2_RESULTS_PATH = GENERATION_DIR / "generation_results_v2.json"
FILTERED_RESULTS_PATH = GENERATION_DIR / "generation_filter_results.json"


def load_v2_metrics() -> Dict:
    if not V2_METRICS_PATH.exists():
        raise FileNotFoundError(f"v2 metrics not found at {V2_METRICS_PATH}")
    return json.loads(V2_METRICS_PATH.read_text())


def load_filtered_metrics() -> Dict:
    if not FILTERED_METRICS_PATH.exists():
        raise FileNotFoundError(f"filtered metrics not found at {FILTERED_METRICS_PATH}")
    return json.loads(FILTERED_METRICS_PATH.read_text())


def compute_comparison(v2_metrics: Dict, filtered_metrics: Dict) -> Dict:
    """
    Produce comparison table for budgets 800 and 1200.

    Returns dict keyed by budget string with:
      v2, filtered, delta_recall, delta_missing, delta_duplicate, delta_unsupported, delta_truncation
    """
    comparison: Dict[str, Dict] = {}
    for budget_str in ("800", "1200"):
        v2 = v2_metrics.get(budget_str, {})
        filt = filtered_metrics.get(budget_str, {})
        if not v2 or not filt:
            continue
        delta_recall = round(filt.get("mean_recall", 0) - v2.get("mean_recall", 0), 6)
        delta_missing = round(filt.get("mean_missing", 0) - v2.get("mean_missing", 0), 6)
        delta_duplicate = round(filt.get("mean_duplicate", 0) - v2.get("mean_duplicate", 0), 6)
        delta_unsupported = round(filt.get("mean_unsupported", 0) - v2.get("mean_unsupported", 0), 6)
        delta_truncation = round(filt.get("truncation_rate", 0) - v2.get("truncation_rate", 0), 6)
        delta_output_tokens = None
        if filt.get("mean_output_tokens") is not None and v2.get("mean_output_tokens") is not None:
            delta_output_tokens = round(filt["mean_output_tokens"] - v2["mean_output_tokens"], 2)
        delta_latency = None
        if filt.get("mean_latency_ms") is not None and v2.get("mean_latency_ms") is not None:
            delta_latency = round(filt["mean_latency_ms"] - v2["mean_latency_ms"], 2)

        comparison[budget_str] = {
            "budget": int(budget_str),
            "v2": v2,
            "filtered": filt,
            "delta_recall": delta_recall,
            "delta_missing": delta_missing,
            "delta_duplicate": delta_duplicate,
            "delta_unsupported": delta_unsupported,
            "delta_truncation_rate": delta_truncation,
            "delta_output_tokens": delta_output_tokens,
            "delta_latency_ms": delta_latency,
        }
    return comparison


def format_comparison_markdown(v2_metrics: Dict, filtered_metrics: Dict) -> str:
    comp = compute_comparison(v2_metrics, filtered_metrics)
    lines = []
    lines.append("| Budget | Condition | Mean Recall | Missing | Duplicates | Unsupported | Truncation |")
    lines.append("|--------|-----------|-------------|---------|------------|-------------|------------|")
    for budget_str in ("800", "1200"):
        if budget_str not in comp:
            continue
        c = comp[budget_str]
        v2 = c["v2"]
        filt = c["filtered"]
        lines.append(
            f"| {budget_str} | v2 baseline | {v2.get('mean_recall', 0):.3f} | {v2.get('mean_missing', 0):.1f} | {v2.get('mean_duplicate', 0):.1f} | {v2.get('mean_unsupported', 0):.1f} | {v2.get('truncation_rate', 0):.2f} |"
        )
        lines.append(
            f"| {budget_str} | filtered | {filt.get('mean_recall', 0):.3f} | {filt.get('mean_missing', 0):.1f} | {filt.get('mean_duplicate', 0):.1f} | {filt.get('mean_unsupported', 0):.1f} | {filt.get('truncation_rate', 0):.2f} |"
        )
        lines.append(
            f"| {budget_str} | delta (filt-baseline) | {c['delta_recall']:+.3f} | {c['delta_missing']:+.1f} | {c['delta_duplicate']:+.1f} | {c['delta_unsupported']:+.1f} | {c['delta_truncation_rate']:+.2f} |"
        )
    return "\n".join(lines)


__all__ = [
    "FILTER_INSTRUCTION",
    "aggregate_by_budget",
    "compute_masked_generation_metrics",
    "build_masked_record_fingerprints",
    "EXPECTED_HYPERTENSION_MRNS",
    "EXPECTED_HYPERTENSION_COUNT",
    "EXPECTED_K",
    "load_v2_metrics",
    "load_filtered_metrics",
    "compute_comparison",
    "format_comparison_markdown",
]
