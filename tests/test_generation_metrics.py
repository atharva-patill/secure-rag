import pytest

from benchmarks.retrieval.generation.generation_metrics import (
    EXPECTED_HYPERTENSION_COUNT,
    EXPECTED_K,
    EXPECTED_HYPERTENSION_MRNS,
    aggregate_by_budget,
    compute_generation_metrics,
    extract_mrns,
    validate_hypertension_ground_truth,
    validate_retrieval_requirement,
)
from benchmarks.retrieval.ground_truth import AGGREGATE_QUERIES_V2


def test_extract_mrns_basic():
    text = "Patients MRN1001 and MRN1005 have hypertension. MRN1104 also."
    mrns = extract_mrns(text)
    assert mrns == ["MRN1001", "MRN1005", "MRN1104"]


def test_extract_mrns_case_insensitive():
    text = "mrn1001 and Mrn1002"
    assert extract_mrns(text) == ["MRN1001", "MRN1002"]


def test_extract_mrns_duplicate_preserved():
    text = "MRN1001 MRN1001 MRN1005"
    assert extract_mrns(text) == ["MRN1001", "MRN1001", "MRN1005"]


def test_extract_mrns_no_match():
    assert extract_mrns("no ids here") == []
    assert extract_mrns("") == []


def test_generation_recall_calculation():
    gt = ["MRN1001", "MRN1005", "MRN1021", "MRN9999"]
    # Answer mentions 2 of 4
    ans = "MRN1001 and MRN1005 are positive."
    m = compute_generation_metrics(ans, gt, retrieved_ids=["MRN1001", "MRN1005", "MRN1021", "MRN9999"])
    assert m["relevant_records_found"] == 2
    assert m["generation_recall"] == pytest.approx(0.5)
    assert m["missing_count"] == 2


def test_missing_record_calculation():
    gt = EXPECTED_HYPERTENSION_MRNS
    # Mention only first 3
    ans = "MRN1001, MRN1005, MRN1021"
    m = compute_generation_metrics(ans, gt, retrieved_ids=gt)
    assert m["missing_count"] == 13
    assert len(m["missing_records"]) == 13
    assert "MRN1104" in m["missing_records"]
    assert "MRN1001" not in m["missing_records"]


def test_duplicate_detection():
    gt = ["MRN1001", "MRN1005"]
    ans = "MRN1001 MRN1001 MRN1001 and MRN1005"
    m = compute_generation_metrics(ans, gt, retrieved_ids=gt)
    # MRN1001 appears 3 times => 2 duplicates
    assert m["duplicate_records"] == 2
    assert m["duplicate_mrns"] == ["MRN1001"]
    # Only one duplicate when appears twice
    ans2 = "MRN1001 MRN1001"
    m2 = compute_generation_metrics(ans2, gt, retrieved_ids=gt)
    assert m2["duplicate_records"] == 1


def test_duplicate_non_gt_not_counted():
    gt = ["MRN1001"]
    ans = "MRN9999 MRN9999 MRN1001"
    m = compute_generation_metrics(ans, gt, retrieved_ids=["MRN1001", "MRN9999"])
    # duplicates of non-GT MRN should not count
    assert m["duplicate_records"] == 0
    assert m["duplicate_mrns"] == []


def test_unsupported_record_detection():
    gt = ["MRN1001", "MRN1005"]
    retrieved = ["MRN1001", "MRN1005"]
    ans = "MRN1001 and MRN9999 and MRN8888"
    m = compute_generation_metrics(ans, gt, retrieved_ids=retrieved)
    assert m["unsupported_count"] == 2
    assert sorted(m["unsupported_records"]) == ["MRN8888", "MRN9999"]


def test_unsupported_empty_when_all_supported():
    gt = ["MRN1001"]
    ans = "MRN1001"
    m = compute_generation_metrics(ans, gt, retrieved_ids=["MRN1001"])
    assert m["unsupported_count"] == 0


def test_deterministic_metric_calculation():
    gt = EXPECTED_HYPERTENSION_MRNS
    ans = "MRN1001 MRN1005 MRN1021 MRN1001"
    retrieved = gt
    m1 = compute_generation_metrics(ans, gt, retrieved_ids=retrieved)
    m2 = compute_generation_metrics(ans, gt, retrieved_ids=retrieved)
    assert m1 == m2
    # Different order but same set should give same recall/missing but ordered differs
    ans_a = "MRN1001 MRN1005"
    ans_b = "MRN1005 MRN1001"
    ma = compute_generation_metrics(ans_a, gt, retrieved_ids=retrieved)
    mb = compute_generation_metrics(ans_b, gt, retrieved_ids=retrieved)
    assert ma["generation_recall"] == mb["generation_recall"]
    assert ma["relevant_records_found"] == mb["relevant_records_found"]
    assert set(ma["missing_records"]) == set(mb["missing_records"])


def test_validation_exactly_16_hypertension_records():
    assert EXPECTED_HYPERTENSION_COUNT == 16
    assert len(EXPECTED_HYPERTENSION_MRNS) == 16
    # Cross-check against ground_truth constant
    agg_hyp = next(q for q in AGGREGATE_QUERIES_V2 if q["qid"] == "AGG_HYPERTENSION")
    assert len(agg_hyp["relevant_records"]) == 16
    assert set(agg_hyp["relevant_records"]) == set(EXPECTED_HYPERTENSION_MRNS)
    # validate helper passes
    issues = validate_hypertension_ground_truth(EXPECTED_HYPERTENSION_MRNS)
    assert any("PASS" in i for i in issues)
    # Fail on wrong count
    bad = EXPECTED_HYPERTENSION_MRNS[:15]
    issues_bad = validate_hypertension_ground_truth(bad)
    assert any("FAIL" in i for i in issues_bad)


def test_validation_experiment_requires_k20_with_16():
    assert EXPECTED_K == 20
    # Pass case
    issues = validate_retrieval_requirement(20, 16)
    assert any("PASS" in i for i in issues)
    # Fail cases
    issues1 = validate_retrieval_requirement(10, 16)
    assert any("FAIL" in i for i in issues1)
    issues2 = validate_retrieval_requirement(20, 10)
    assert any("FAIL" in i for i in issues2)
    issues3 = validate_retrieval_requirement(20, 15)
    assert any("FAIL" in i for i in issues3)


def test_ground_truth_hypertension_matches_spec():
    # Spec says ground_truth_v2 has 16 for "Give me all patients with hypertension." via AGG_HYPERTENSION
    q = next(x for x in AGGREGATE_QUERIES_V2 if x["qid"] == "AGG_HYPERTENSION")
    assert q["question"] == "Which patients have Hypertension?"
    # The canonical query string for experiment is different but maps to same GT set
    assert len(q["relevant_records"]) == 16


def test_aggregate_by_budget():
    # Simulate 2 budgets with known recalls
    results = [
        {"generation_budget": 200, "generation_recall": 0.5, "missing_count": 8, "duplicate_records": 0, "unsupported_count": 0, "truncation": False, "latency_ms": 100, "output_tokens": 150},
        {"generation_budget": 200, "generation_recall": 0.6, "missing_count": 6, "duplicate_records": 1, "unsupported_count": 0, "truncation": True, "latency_ms": 110, "output_tokens": 160},
        {"generation_budget": 400, "generation_recall": 1.0, "missing_count": 0, "duplicate_records": 0, "unsupported_count": 0, "truncation": False, "latency_ms": 200, "output_tokens": 300},
    ]
    agg = aggregate_by_budget(results)
    assert "200" in agg
    assert "400" in agg
    assert agg["200"]["mean_recall"] == pytest.approx(0.55)
    assert agg["200"]["runs"] == 2
    assert agg["200"]["truncation_rate"] == pytest.approx(0.5)
    assert agg["400"]["mean_recall"] == pytest.approx(1.0)


def test_full_hypertension_recall():
    # Mention all 16
    ans = " ".join(EXPECTED_HYPERTENSION_MRNS)
    m = compute_generation_metrics(ans, EXPECTED_HYPERTENSION_MRNS, retrieved_ids=EXPECTED_HYPERTENSION_MRNS)
    assert m["relevant_records_found"] == 16
    assert m["generation_recall"] == pytest.approx(1.0)
    assert m["missing_count"] == 0


def test_no_mentions():
    gt = EXPECTED_HYPERTENSION_MRNS
    m = compute_generation_metrics("I don't know.", gt, retrieved_ids=gt)
    assert m["relevant_records_found"] == 0
    assert m["generation_recall"] == 0.0
    assert m["missing_count"] == 16
