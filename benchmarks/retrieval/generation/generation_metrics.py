"""Metrics for the controlled generation-budget experiment.

Pure, deterministic helpers that do NOT require Ollama or external APIs.

v1: MRN-string recall (invalid for masked context – guarantees 0 when masking MRNs).
v2: Masked-record coverage – deterministic mapping outside LLM:
     masked chunk index -> ground-truth MRN, and answer coverage is evaluated
     via stable masked attributes (age, diagnosis, treatment, admission
     content) that remain visible to the LLM, NOT raw MRN strings.

The v2 measurand is "enumerated patient/record coverage" as described
in the task: we count how many of the 16 relevant hypertension masked
records are represented in the generated answer, plus duplicate/
unsupported distinctions, without requiring raw MRN strings.
"""

import re
from collections import Counter
from typing import Dict, List, Set, Tuple

# Canonical ground truth for the hypertension aggregate query.
# Must match benchmarks/retrieval/ground_truth.py:AGGREGATE_QUERIES_V2[AGG_HYPERTENSION]
EXPECTED_HYPERTENSION_MRNS: List[str] = [
    "MRN1001",
    "MRN1005",
    "MRN1021",
    "MRN1025",
    "MRN1026",
    "MRN1052",
    "MRN1059",
    "MRN1066",
    "MRN1070",
    "MRN1074",
    "MRN1085",
    "MRN1104",
    "MRN1107",
    "MRN1111",
    "MRN1118",
    "MRN1119",
]

EXPECTED_HYPERTENSION_COUNT = 16
EXPECTED_K = 20

# Regex for MRN extraction. Clinical records use MRNxxxx where xxxx are digits.
_MRN_RE = re.compile(r"\bMRN\d+\b", re.IGNORECASE)


def extract_mrns(text: str) -> List[str]:
    """Extract MRN mentions in order of appearance, normalized to upper-case.

    Duplicate mentions are preserved for duplicate detection.
    """
    if not text:
        return []
    raw = _MRN_RE.findall(text)
    return [m.upper() for m in raw]


def extract_mrn_set(text: str) -> Set[str]:
    """Extract deduplicated set of MRNs."""
    return set(extract_mrns(text))


# ---------------------------------------------------------------------------
# v2 – masked-record matching utilities
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _extract_age(text: str) -> str:
    m = re.search(r"(\d+)-year-old\s+(male|female)", text, re.I)
    return m.group(0).lower() if m else ""


def _extract_age_num(text: str) -> str:
    m = re.search(r"(\d+)-year-old", text, re.I)
    return m.group(0).lower() if m else ""


def _extract_diagnosis(text: str) -> str:
    m = re.search(r"Diagnosis:\s*([^\n]+)", text, re.I)
    if m:
        # strip trailing period segment but keep full line for uniqueness
        return m.group(1).strip()
    return ""


def _extract_treatment(text: str) -> str:
    m = re.search(r"Treatment:\s*([^\n]+)", text, re.I)
    if m:
        return m.group(1).strip()
    return ""


def _extract_admission_reason(text: str) -> str:
    # First sentence after age: "was admitted ... with <reason>."
    m = re.search(r"was admitted[^\n]*?with\s+([^\.\n]+)", text, re.I)
    return m.group(1).strip() if m else ""


def build_masked_record_fingerprints(
    retrieved_chunks: List[str], retrieved_ids: List[str]
) -> List[Dict]:
    """Build per-record fingerprints from masked chunks.

    Each fingerprint is deterministic and auditable, derived only from
    content visible to the LLM (masked chunks). No raw MRN is used.
    """
    fps: List[Dict] = []
    for chunk, rid in zip(retrieved_chunks, retrieved_ids):
        age = _extract_age(chunk)
        age_num = _extract_age_num(chunk)
        diagnosis = _extract_diagnosis(chunk)
        treatment = _extract_treatment(chunk)
        admission = _extract_admission_reason(chunk)
        fps.append(
            {
                "record_id": rid,
                "masked_chunk": chunk,
                "norm_chunk": _normalize(chunk),
                "age": _normalize(age) if age else "",
                "age_num": _normalize(age_num) if age_num else "",
                "diagnosis": diagnosis,
                "norm_diagnosis": _normalize(diagnosis),
                "treatment": treatment,
                "norm_treatment": _normalize(treatment),
                "admission": admission,
                "norm_admission": _normalize(admission),
            }
        )
    return fps


def _split_answer_into_patient_segments(answer_text: str) -> List[str]:
    """Split answer into enumerated patient blocks.

    LLM enumerates patients as:
      1. [NAME_MASKED], a 53-year-old ...
      2. [NAME_MASKED], a 43-year-old ...

    We split on numbered list markers and retain blocks containing age phrase.
    If no numbered markers found, fall back to searching for age phrases as
    separate pseudo-segments.
    """
    if not answer_text:
        return []
    # Primary: split on numbered list "1.", "2.", etc.
    # Prefix newline to catch leading "1." without preceding newline.
    parts = re.split(r"\n\s*\d+\.\s*", "\n" + answer_text)
    segments: List[str] = []
    for p in parts:
        if re.search(r"\d+-year-old", p, re.I):
            segments.append(p.strip())
    if segments:
        return segments
    # Fallback: no enumerated list – treat each age mention as segment
    # Extract windows around each age mention
    age_iters = list(re.finditer(r"\d+-year-old\s+(?:male|female)[^\n]*", answer_text, re.I))
    if not age_iters:
        # also try without gender (hallucinated)
        age_iters = list(re.finditer(r"\d+-year-old[^\n]*", answer_text, re.I))
    for m in age_iters:
        # take a window of ~300 chars starting at match for context
        start = max(0, m.start() - 20)
        end = min(len(answer_text), m.end() + 280)
        segments.append(answer_text[start:end].strip())
    # Deduplicate exact duplicates while preserving order
    seen = set()
    uniq: List[str] = []
    for s in segments:
        ns = _normalize(s)
        if ns not in seen:
            seen.add(ns)
            uniq.append(s)
    return uniq if uniq else segments


def _score_segment_against_fingerprint(segment_norm: str, fp: Dict) -> int:
    """Deterministic score for how well a segment matches a masked record fingerprint.

    Higher = better. Must have age_num at least. Scoring is auditable:
      +10 age_num present (mandatory)
      +2  exact age+gender present
      +5  first diagnosis term present
      +3  full diagnosis snippet present (first 40 chars)
      +5  secondary diagnosis term(s) present (for records with multiple diagnoses)
      +2  first treatment drug present
      +2  admission reason phrase present
    """
    # Mandatory: age_num must be present
    if fp["age_num"] and fp["age_num"] not in segment_norm:
        return 0
    if not fp["age_num"]:
        return 0
    score = 10
    if fp["age"] and fp["age"] in segment_norm:
        score += 2
    # Diagnosis: at least first term should be present for hypertension records
    diag_terms = [t.strip().lower() for t in fp["norm_diagnosis"].split(";")]
    # Filter empty
    diag_terms = [t for t in diag_terms if t]
    if diag_terms:
        first_term = diag_terms[0].split(",")[0].strip()
        # For "hypertension" records, first_term is "hypertension"
        if first_term and first_term in segment_norm:
            score += 5
            # Bonus for longer diagnosis snippet
            if fp["norm_diagnosis"][:40].strip() and fp["norm_diagnosis"][:40].strip() in segment_norm:
                score += 3
        else:
            # No diagnosis term -> still possible if admission contains hypertension description?
            # e.g., segment may say "with persistent hypertension" which is not in diagnosis line extraction but still indicates hypertension
            # Check admission reason contains hypertension word
            if "hypertension" in segment_norm and "hypertension" in fp["norm_diagnosis"]:
                score += 3
            else:
                # penalize but not zero – age alone not enough to claim coverage if diagnosis missing
                score -= 2
        # Secondary terms (distinctive like "iron deficiency anaemia")
        for term in diag_terms[1:]:
            base = term.split(",")[0].strip()
            if base and base in segment_norm:
                score += 5
            # Also check presence of full term
            if term in segment_norm:
                score += 1
        # Special handling: for MRN1104, distinctive "iron deficiency anaemia" must be present to distinguish from MRN1107
        # Our scoring already gives +5 for that secondary term
    # Treatment: check first drug
    if fp["norm_treatment"]:
        treat_first = fp["norm_treatment"].split(",")[0].strip()
        if treat_first and treat_first in segment_norm:
            score += 2
        # Also check if any treatment token appears (e.g., "telmisartan" )
        # Already covered
    # Admission reason
    if fp["norm_admission"] and fp["norm_admission"][:30].strip() in segment_norm:
        score += 2
    # Also check for distinctive notes phrase: e.g., "fibromyalgia" appears both in diagnosis and notes/admission
    # Already handled via diag_terms
    return max(score, 0)


def _match_segments_to_records(
    answer_text: str, fingerprints: List[Dict]
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """Match each answer segment to best fingerprint.

    Returns (matched_ids_in_order, matched_relevant_ids, counter_by_id)
    matched_ids_in_order includes every segment's best match (or None if no match).
    """
    segments = _split_answer_into_patient_segments(answer_text)
    norm_segments = [_normalize(s) for s in segments]
    matched_ordered: List[str] = []
    counter: Dict[str, int] = Counter()
    for seg_norm in norm_segments:
        best_id = None
        best_score = 0
        best_fp = None
        for fp in fingerprints:
            sc = _score_segment_against_fingerprint(seg_norm, fp)
            if sc > best_score:
                best_score = sc
                best_id = fp["record_id"]
                best_fp = fp
        # Threshold: must be at least age_num + diagnosis term => ~15
        # Lower threshold to 10 to allow truncated bullets with only age+admission
        # But to avoid false positives, require >=12
        if best_id is not None and best_score >= 12:
            matched_ordered.append(best_id)
            counter[best_id] += 1
        else:
            # Check if segment contains age_num but no good fingerprint -> hallucinated/unsupported
            # We still consider it as an unsupported mention if it looks like a patient mention
            # Mark with special hallucination id if age present but no fingerprint matched
            # For now, skip (not counted as matched), but caller can count segments as attempted mentions
            # We record as None for debugging but not in matched list
            # To distinguish unsupported vs missing, we need to know segments that looked like patients but matched none
            # We'll append a sentinel for counting
            if re.search(r"\d+-year-old", seg_norm):
                matched_ordered.append(f"__HALLUCINATED__:{seg_norm[:30]}")
            else:
                matched_ordered.append("__NO_MATCH__")
    # Filter to only real matched ids
    real_matched = [m for m in matched_ordered if not m.startswith("__")]
    return matched_ordered, real_matched, dict(counter)


def compute_masked_generation_metrics(
    answer_text: str,
    retrieved_chunks: List[str],
    retrieved_ids: List[str],
    ground_truth_mrns: List[str] | Set[str],
    k: int = 20,
) -> Dict:
    """Compute generation coverage using masked-record fingerprints (v2).

    This metric does NOT require raw MRN strings in the answer.
    It evaluates whether the answer represents the relevant masked records
    using information actually available in the masked context (age,
    diagnosis, treatment, admission content).

    Args:
        answer_text: Raw LLM answer.
        retrieved_chunks: Masked context chunks in retrieval order (len K).
        retrieved_ids: Parallel MRN ids for each chunk (deterministic mapping
                       chunk index -> MRN, built outside LLM).
        ground_truth_mrns: 16 hypertension MRNs expected.
        k: expected K (for validation, default 20).

    Returns:
        dict with:
            relevant_records_found, generation_recall, missing_records,
            duplicate_records, duplicate_mrns, unsupported_records,
            found_mrns, found_mrns_ordered, etc., plus
            enumerated_patient_count, hallucinated_count, segment_details
        All deterministic, auditable.
    """
    gt_set = set(ground_truth_mrns)
    retrieved_set = set(retrieved_ids) if retrieved_ids is not None else set()
    fingerprints = build_masked_record_fingerprints(retrieved_chunks, retrieved_ids)

    # Match segments to records
    matched_ordered_all, real_matched, counter = _match_segments_to_records(answer_text, fingerprints)
    # Deduplicate matched set
    matched_set = set(real_matched)
    # Relevant found = intersection with ground truth
    relevant_found_set = matched_set.intersection(gt_set)
    relevant_records_found = len(relevant_found_set)
    generation_recall = relevant_records_found / len(gt_set) if gt_set else 0.0
    missing_records = sorted(gt_set - relevant_found_set)
    # Duplicate detection: extra mentions beyond first for GT MRNs
    duplicate_records = sum(count - 1 for mrn, count in counter.items() if mrn in gt_set and count > 1)
    duplicate_mrns = sorted([mrn for mrn, count in counter.items() if mrn in gt_set and count > 1])
    # Unsupported: matched records that are in retrieved but NOT in ground truth
    # plus hallucinated patient-like segments that matched nothing
    unsupported_matched = matched_set - gt_set
    # Only those that are in retrieved context are "unsupported non-relevant"
    # Hallucinated are also unsupported but not in retrieved
    # For this metric, unsupported = matched non-relevant retrieved OR hallucinated patient segments
    hallucinated_count = sum(1 for m in matched_ordered_all if m.startswith("__HALLUCINATED__"))
    unsupported_records = sorted(unsupported_matched.intersection(retrieved_set))
    unsupported_count = len(unsupported_records) + hallucinated_count
    # Found lists
    found_gt_ordered: List[str] = []
    seen = set()
    for m in real_matched:
        if m in gt_set and m not in seen:
            seen.add(m)
            found_gt_ordered.append(m)
    # Enumerated patient count = number of segments that looked like patients (age present)
    enumerated_patient_count = sum(1 for m in matched_ordered_all if not m == "__NO_MATCH__")
    # For audit: provide segment fingerprints
    return {
        "relevant_records_found": relevant_records_found,
        "generation_recall": generation_recall,
        "missing_records": missing_records,
        "missing_count": len(missing_records),
        "duplicate_records": duplicate_records,
        "duplicate_mrns": duplicate_mrns,
        "unsupported_records": unsupported_records,
        "unsupported_count": unsupported_count,
        "found_mrns": sorted(matched_set),
        "found_mrns_ordered": real_matched,
        "found_gt_ordered": found_gt_ordered,
        "ground_truth_count": len(gt_set),
        "enumerated_patient_count": enumerated_patient_count,
        "hallucinated_count": hallucinated_count,
        "matched_ordered_all": matched_ordered_all,
        "segment_count": len(_split_answer_into_patient_segments(answer_text)),
        "counter_by_id": counter,
        "measurand": "enumerated patient/record coverage via deterministic masked-record fingerprint matching (age + diagnosis + treatment + admission) without requiring raw MRN strings; recall = relevant masked records represented in answer / 16",
        "measurand_details": "Each retrieved masked chunk (age, diagnosis, treatment, admission reason) is fingerprinted. Answer is split into enumerated patient segments (numbered list). Each segment is scored against all fingerprints using deterministic age+diagnosis+treatment overlap; best score >=12 is assigned to that record. This distinguishes relevant vs non-relevant vs duplicate without exposing raw MRNs.",
    }


# Keep original MRN-based metric for legacy / tests but flag as invalid for masked context
def compute_generation_metrics(
    answer_text: str,
    ground_truth_mrns: List[str] | Set[str],
    retrieved_ids: List[str] | Set[str] | None = None,
) -> Dict:
    """Compute record-level generation metrics for one answer (legacy MRN string matching).

    NOTE: This metric requires raw MRN strings in the answer and is INVALID
    when masking replaces MRNs with [PATIENT_ID_MASKED]. Use
    compute_masked_generation_metrics for masked contexts.
    """
    gt_set = set(ground_truth_mrns)
    found_ordered = extract_mrns(answer_text)
    found_set = set(found_ordered)
    retrieved_set = set(retrieved_ids) if retrieved_ids is not None else None
    relevant_found_set = found_set.intersection(gt_set)
    relevant_records_found = len(relevant_found_set)
    generation_recall = relevant_records_found / len(gt_set) if gt_set else 0.0
    missing_records = sorted(gt_set - relevant_found_set)
    counter = Counter(found_ordered)
    duplicate_records = sum(
        count - 1 for mrn, count in counter.items() if mrn in gt_set and count > 1
    )
    duplicate_mrns = sorted([mrn for mrn, count in counter.items() if mrn in gt_set and count > 1])
    unsupported_records: List[str] = []
    unsupported_count = 0
    if retrieved_set is not None:
        unsupported_set = found_set - retrieved_set
        unsupported_records = sorted(unsupported_set)
        unsupported_count = len(unsupported_records)
    found_gt_ordered = []
    seen = set()
    for m in found_ordered:
        if m in gt_set and m not in seen:
            seen.add(m)
            found_gt_ordered.append(m)
    return {
        "relevant_records_found": relevant_records_found,
        "generation_recall": generation_recall,
        "missing_records": missing_records,
        "missing_count": len(missing_records),
        "duplicate_records": duplicate_records,
        "duplicate_mrns": duplicate_mrns,
        "unsupported_records": unsupported_records,
        "unsupported_count": unsupported_count,
        "found_mrns": sorted(found_set),
        "found_mrns_ordered": found_ordered,
        "found_gt_ordered": found_gt_ordered,
        "ground_truth_count": len(gt_set),
    }


def validate_hypertension_ground_truth(ground_truth_mrns: List[str] | Set[str]) -> List[str]:
    """Validate that exactly 16 hypertension ground-truth records are expected."""
    issues: List[str] = []
    mrns = list(ground_truth_mrns)
    if len(mrns) != EXPECTED_HYPERTENSION_COUNT:
        issues.append(f"FAIL: Expected {EXPECTED_HYPERTENSION_COUNT} hypertension records, got {len(mrns)}")
    if len(set(mrns)) != len(mrns):
        issues.append("FAIL: Duplicate MRNs in hypertension ground truth")
    if set(mrns) != set(EXPECTED_HYPERTENSION_MRNS):
        issues.append(f"FAIL: Hypertension MRNs mismatch. Expected {EXPECTED_HYPERTENSION_MRNS}, got {sorted(mrns)}")
    if not issues:
        issues.append(f"PASS: {EXPECTED_HYPERTENSION_COUNT} hypertension records validated")
    return issues


def validate_retrieval_requirement(retrieved_count: int, relevant_count: int) -> List[str]:
    """Validate that K=20 retrieval returned 16 relevant records."""
    issues: List[str] = []
    if retrieved_count != EXPECTED_K:
        issues.append(f"FAIL: Expected retrieved_count == {EXPECTED_K}, got {retrieved_count}")
    if relevant_count != EXPECTED_HYPERTENSION_COUNT:
        issues.append(
            f"FAIL: Expected relevant_count == {EXPECTED_HYPERTENSION_COUNT}, got {relevant_count}"
        )
    if retrieved_count == EXPECTED_K and relevant_count == EXPECTED_HYPERTENSION_COUNT:
        issues.append(f"PASS: K={EXPECTED_K} retrieved {relevant_count}/{EXPECTED_HYPERTENSION_COUNT} relevant records")
    return issues


def aggregate_by_budget(results: List[Dict]) -> Dict:
    """Aggregate generation metrics by generation budget.

    Handles tri-state truncation: True (truncated), False (not truncated), None/"unknown".
    """
    import math

    by_budget: Dict[int, List[Dict]] = {}
    for r in results:
        b = r.get("generation_budget")
        by_budget.setdefault(b, []).append(r)

    aggregated: Dict[str, Dict] = {}
    for budget, runs in sorted(by_budget.items()):
        recalls = [x.get("generation_recall", 0.0) for x in runs]
        missings = [x.get("missing_count", 0) for x in runs]
        dups = [x.get("duplicate_records", 0) for x in runs]
        unsupp = [x.get("unsupported_count", 0) for x in runs]
        # Truncation handling: True/False/None or "unknown"
        trunc_vals = [x.get("truncation") for x in runs]
        # Normalize: True=1, False=0, unknown/None -> not counted in truncation_rate but counted separately
        trunc_known = [1 if v is True else 0 if v is False else None for v in trunc_vals]
        trunc_known_filtered = [v for v in trunc_known if v is not None]
        unknown_count = sum(1 for v in trunc_known if v is None)
        lat = [x.get("latency_ms", 0) for x in runs if x.get("latency_ms") is not None]
        toks = [x.get("output_tokens", 0) for x in runs if x.get("output_tokens") is not None]
        prompt_toks = [x.get("prompt_tokens", 0) for x in runs if x.get("prompt_tokens") is not None]

        def _mean(vals):
            return sum(vals) / len(vals) if vals else 0.0

        def _std(vals):
            if not vals:
                return 0.0
            m = _mean(vals)
            var = sum((v - m) ** 2 for v in vals) / len(vals)
            return math.sqrt(var)

        aggregated[str(budget)] = {
            "budget": budget,
            "runs": len(runs),
            "mean_recall": round(_mean(recalls), 6),
            "std_recall": round(_std(recalls), 6),
            "min_recall": round(min(recalls), 6) if recalls else 0.0,
            "max_recall": round(max(recalls), 6) if recalls else 0.0,
            "mean_missing": round(_mean(missings), 6),
            "std_missing": round(_std(missings), 6),
            "mean_duplicate": round(_mean(dups), 6),
            "mean_unsupported": round(_mean(unsupp), 6),
            "truncation_rate": round(_mean(trunc_known_filtered), 6) if trunc_known_filtered else 0.0,
            "truncation_count": sum(v for v in trunc_known_filtered if v),
            "unknown_termination_count": unknown_count,
            "unknown_termination_rate": round(unknown_count / len(runs), 6) if runs else 0.0,
            "mean_latency_ms": round(_mean(lat), 2) if lat else None,
            "mean_output_tokens": round(_mean(toks), 2) if toks else None,
            "mean_prompt_tokens": round(_mean(prompt_toks), 2) if prompt_toks else None,
        }
    return aggregated
