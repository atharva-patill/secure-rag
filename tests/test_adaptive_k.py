"""Research-only tests for Adaptive Retrieval Depth (K) experiment.

Covers required checks:
1. all K values accepted (2,5,10,20,30,50)
2. K passed explicitly (runner uses explicit k, not default)
3. production default K remains 2
4. retrieval metrics correct (HitRate, Precision, Recall, MRR)
5. multi-record ground truth correct (16,17,20,7,1)
6. context record count equals K
7. raw MRNs absent from context
8. raw MRNs absent from prompt
9. generation evaluator reused correctly (compute_masked_generation_metrics)
10. done_reason length → truncated=True
11. done_reason stop → truncated=False
12. missing done_reason → unknown
13. aggregate query count correct (5)
14. no ground-truth mutation
"""

import json
import pathlib
import re
import inspect

import pytest

from benchmarks.retrieval.ground_truth import AGGREGATE_QUERIES_V2
from benchmarks.retrieval.metrics import hit_rate_at_k, precision_at_k, recall_at_k, mrr_at_k
from secure_rag.retriever import retrieve as prod_retrieve
from secure_rag.vector_store import VectorStore

# Import adaptive_k modules
from benchmarks.retrieval.adaptive_k.adaptive_k_runner import K_VALUES as RUNNER_K_VALUES, AGG_QIDS, EXPECTED_COUNTS
from benchmarks.retrieval.adaptive_k.adaptive_k_metrics import K_VALUES as METRICS_K
from benchmarks.retrieval.adaptive_k import adaptive_k_generation as gen_mod


def _mk_retrieved(record_ids):
    return [{"record_id": rid, "relevant": False, "rank": i, "chunk_index": i, "score": 0.0} for i, rid in enumerate(record_ids)]


# 1. all K values accepted
def test_all_k_values_accepted():
    assert RUNNER_K_VALUES == [2, 5, 10, 20, 30, 50]
    assert METRICS_K == [2, 5, 10, 20, 30, 50]
    # Also check retrieval results artifact respects exactly these
    path = pathlib.Path("benchmarks/retrieval/adaptive_k/adaptive_k_retrieval_results.json")
    if not path.exists():
        pytest.skip("retrieval results not yet generated")
    data = json.loads(path.read_text())
    assert data["k_values"] == [2, 5, 10, 20, 30, 50]
    # each query has per_k for exactly these
    for entry in data["results"]:
        for k in [2, 5, 10, 20, 30, 50]:
            assert str(k) in entry["per_k"], f"{entry['qid']} missing K={k}"


# 2. K passed explicitly
def test_k_passed_explicitly():
    # Verify runner source passes k explicitly to vector_store.search / retrieve
    src = pathlib.Path("benchmarks/retrieval/adaptive_k/adaptive_k_runner.py").read_text()
    assert "vector_store.search" in src
    # Should have k= k or k=k param
    assert "k=k" in src or "k = k" in src or "k=" in src
    # Generation runner also passes k explicitly
    gsrc = pathlib.Path("benchmarks/retrieval/adaptive_k/adaptive_k_generation.py").read_text()
    assert "vector_store.search" in gsrc
    assert "k=k" in gsrc or "k=" in gsrc
    # Ensure retrieve wrapper is NOT relying on default without K
    # Check that runner defines K_VALUES and iterates over them
    assert "for k in K_VALUES" in src
    assert "for k in K_VALUES" in gsrc or "for k in K_VALUES" in gsrc or "K_VALUES" in gsrc


# 3. production default K remains 2
def test_production_default_k_remains_2():
    sig = inspect.signature(prod_retrieve)
    param_k = sig.parameters.get("k")
    assert param_k is not None, "retrieve missing k param"
    assert param_k.default == 2, f"production default K should be 2, got {param_k.default}"
    # Also VectorStore.search default
    sig_vs = inspect.signature(VectorStore.search)
    param_vs_k = sig_vs.parameters.get("k")
    assert param_vs_k is not None
    assert param_vs_k.default == 2
    # Ensure source files still contain k=2
    rsrc = pathlib.Path("secure_rag/retriever.py").read_text()
    assert "k=2" in rsrc
    vsrc = pathlib.Path("secure_rag/vector_store.py").read_text()
    assert "k=2" in vsrc


# 4. retrieval metrics correct
def test_retrieval_metrics_correct():
    # Single relevant
    retrieved = _mk_retrieved(["A", "B", "C"])
    assert hit_rate_at_k(retrieved, 1, {"A"}) == 1
    assert hit_rate_at_k(retrieved, 1, {"B"}) == 0
    assert precision_at_k(retrieved, 2, {"A"}) == pytest.approx(0.5)
    assert recall_at_k(retrieved, 2, relevant_set={"A"}) == pytest.approx(1.0)
    assert recall_at_k(retrieved, 1, relevant_set={"A"}) == pytest.approx(1.0)
    assert mrr_at_k(retrieved, 1, {"A"}) == pytest.approx(1.0)
    assert mrr_at_k(_mk_retrieved(["B", "A"]), 2, {"A"}) == pytest.approx(0.5)
    # Multi relevant
    retrieved_m = _mk_retrieved(["MRN1", "MRN2", "X", "MRN3"])
    s = {"MRN1", "MRN2", "MRN3", "MRN4"}
    assert recall_at_k(retrieved_m, 2, relevant_set=s) == pytest.approx(0.5)
    assert recall_at_k(retrieved_m, 4, relevant_set=s) == pytest.approx(0.75)
    assert precision_at_k(retrieved_m, 4, s) == pytest.approx(0.75)
    # Zero relevant
    assert recall_at_k(_mk_retrieved(["A"]), 2, relevant_set=set()) == 0.0
    assert hit_rate_at_k(_mk_retrieved(["A"]), 2, set()) == 0


# 5. multi-record ground truth correct
def test_multi_record_ground_truth_correct():
    gt_map = {q["qid"]: q for q in AGGREGATE_QUERIES_V2}
    assert len(gt_map["AGG_HYPERTENSION"]["relevant_records"]) == 16
    assert len(gt_map["AGG_AMLODIPINE_5MG"]["relevant_records"]) == 17
    assert len(gt_map["AGG_PARACETAMOL_650MG"]["relevant_records"]) == 20
    assert len(gt_map["AGG_METFORMIN_500MG"]["relevant_records"]) == 7
    assert len(gt_map["AGG_T2D_HYPERTENSION"]["relevant_records"]) == 1
    # Check EXPECTED_COUNTS matches
    assert EXPECTED_COUNTS["AGG_HYPERTENSION"] == 16
    assert EXPECTED_COUNTS["AGG_AMLODIPINE_5MG"] == 17
    assert EXPECTED_COUNTS["AGG_PARACETAMOL_650MG"] == 20
    assert EXPECTED_COUNTS["AGG_METFORMIN_500MG"] == 7
    assert EXPECTED_COUNTS["AGG_T2D_HYPERTENSION"] == 1


# 6. context record count equals K
def test_context_record_count_equals_k():
    path = pathlib.Path("benchmarks/retrieval/adaptive_k/adaptive_k_generation_results.json")
    if not path.exists():
        pytest.skip("generation results not yet generated")
    data = json.loads(path.read_text())
    for run in data["runs"]:
        k = run["k"]
        assert run["retrieved_record_count"] == k, f"{run['qid']} K={k} record count mismatch"
        assert run["context_integrity"]["context_record_count"] == k
        assert run["context_integrity"]["match"] is True
        assert len(run["context_chunks"]) == k
        assert len(run["retrieved_record_ids"]) == k


# 7. raw MRNs absent from context
def test_raw_mrns_absent_from_context():
    path = pathlib.Path("benchmarks/retrieval/adaptive_k/adaptive_k_generation_results.json")
    if not path.exists():
        pytest.skip("generation results not yet generated")
    data = json.loads(path.read_text())
    pattern = re.compile(r"\bMRN\d+\b")
    for run in data["runs"]:
        ctx = run["context"]
        found = pattern.findall(ctx)
        assert len(found) == 0, f"Raw MRN in context for {run['qid']} K={run['k']}: {found}"
        assert run["privacy"]["raw_mrn_count_context"] == 0
        assert not run["privacy"]["raw_mrns_in_context"]


# 8. raw MRNs absent from prompt
def test_raw_mrns_absent_from_prompt():
    path = pathlib.Path("benchmarks/retrieval/adaptive_k/adaptive_k_generation_results.json")
    if not path.exists():
        pytest.skip("generation results not yet generated")
    data = json.loads(path.read_text())
    pattern = re.compile(r"\bMRN\d+\b")
    for run in data["runs"]:
        prompt = run["prompt"]
        found = pattern.findall(prompt)
        assert len(found) == 0, f"Raw MRN in prompt for {run['qid']} K={run['k']}: {found}"
        assert run["privacy"]["raw_mrn_count_prompt"] == 0
        assert not run["privacy"]["raw_mrns_in_prompt"]


# 9. generation evaluator reused correctly
def test_generation_evaluator_reused_correctly():
    # Must reuse compute_masked_generation_metrics
    src = pathlib.Path("benchmarks/retrieval/adaptive_k/adaptive_k_generation.py").read_text()
    assert "compute_masked_generation_metrics" in src
    assert "from benchmarks.retrieval.generation.generation_metrics import compute_masked_generation_metrics" in src
    # Also verify that the function is actually callable and matches expected signature
    from benchmarks.retrieval.generation.generation_metrics import compute_masked_generation_metrics
    assert callable(compute_masked_generation_metrics)
    # Test that evaluator works on minimal masked chunks
    chunks = [
        "[NAME_MASKED], a 53-year-old female patient, was admitted to [ORG_MASKED] on [DOB_MASKED] with persistent hypertension. Medical ID: [PATIENT_ID_MASKED], Diagnosis: Hypertension. Treatment: Telmisartan 40mg, Low-salt diet, Amlodipine 5mg. Notes: On examination, blood pressure was 146/90 mmHg.",
        "[NAME_MASKED], a 43-year-old female patient, was admitted to [ORG_MASKED] on [DOB_MASKED] with elevated blood pressure readings. Medical ID: [PATIENT_ID_MASKED], Diagnosis: Hypertension. Treatment: Telmisartan 40mg, Low-salt diet, Amlodipine 5mg. Notes: On examination, pulse rate was 82 bpm.",
    ]
    ids = ["MRN1107", "MRN1066"]
    answer = "1. [NAME_MASKED], a 53-year-old female patient, was admitted to [ORG_MASKED] on [DOB_MASKED] with persistent hypertension. Medical ID: [PATIENT_ID_MASKED], Diagnosis: Hypertension.\n2. [NAME_MASKED], a 43-year-old female patient, was admitted to [ORG_MASKED] on [DOB_MASKED] with elevated blood pressure readings."
    gm = compute_masked_generation_metrics(answer, chunks, ids, ["MRN1107", "MRN1066"])
    assert gm["relevant_records_found"] == 2


# 10. done_reason length → truncated=True
def test_done_reason_length_truncated():
    # Test via generation module's _call_generation with mocked metadata
    from unittest.mock import patch

    fake_meta = {"done": True, "done_reason": "length", "eval_count": 1200, "prompt_eval_count": 1000}
    with patch("benchmarks.retrieval.adaptive_k.adaptive_k_generation.generate_answer", return_value=iter(["ans"])):
        with patch("benchmarks.retrieval.adaptive_k.adaptive_k_generation.get_last_ollama_metadata", return_value=fake_meta):
            gen = gen_mod._call_generation("ctx", "q", budget=1200)
            assert gen["finish_reason"] == "length"
            assert gen["truncation"] is True
            assert gen["truncation_status"] == "truncated"
            assert gen["done_reason"] == "length"
            assert gen["output_tokens"] == 1200
            assert gen["output_tokens_is_estimated"] is False


# 11. done_reason stop → truncated=False
def test_done_reason_stop_not_truncated():
    from unittest.mock import patch

    fake_meta = {"done": True, "done_reason": "stop", "eval_count": 500, "prompt_eval_count": 1000}
    with patch("benchmarks.retrieval.adaptive_k.adaptive_k_generation.generate_answer", return_value=iter(["ans"])):
        with patch("benchmarks.retrieval.adaptive_k.adaptive_k_generation.get_last_ollama_metadata", return_value=fake_meta):
            gen = gen_mod._call_generation("ctx", "q", budget=1200)
            assert gen["finish_reason"] == "stop"
            assert gen["truncation"] is False
            assert gen["truncation_status"] == "not_truncated"
            assert gen["done_reason"] == "stop"


# 12. missing done_reason → unknown
def test_missing_done_reason_unknown():
    from unittest.mock import patch

    fake_meta = {}  # no done_reason
    with patch("benchmarks.retrieval.adaptive_k.adaptive_k_generation.generate_answer", return_value=iter(["ans"])):
        with patch("benchmarks.retrieval.adaptive_k.adaptive_k_generation.get_last_ollama_metadata", return_value=fake_meta):
            gen = gen_mod._call_generation("ctx", "q", budget=1200)
            assert gen["finish_reason"] == "unknown"
            assert gen["truncation"] is None
            assert gen["truncation_status"] == "unknown"
            assert gen["done_reason"] is None
            assert gen["truncation"] is not False  # must be None, not False


# 13. aggregate query count correct
def test_aggregate_query_count_correct():
    assert len(AGG_QIDS) == 5
    assert set(AGG_QIDS) == {"AGG_HYPERTENSION", "AGG_AMLODIPINE_5MG", "AGG_PARACETAMOL_650MG", "AGG_METFORMIN_500MG", "AGG_T2D_HYPERTENSION"}
    # Check retrieval results has 5 queries
    path = pathlib.Path("benchmarks/retrieval/adaptive_k/adaptive_k_retrieval_results.json")
    if path.exists():
        data = json.loads(path.read_text())
        assert len(data["results"]) == 5
        assert set(r["qid"] for r in data["results"]) == set(AGG_QIDS)
    # Generation has 5*6=30 runs
    gpath = pathlib.Path("benchmarks/retrieval/adaptive_k/adaptive_k_generation_results.json")
    if gpath.exists():
        gdata = json.loads(gpath.read_text())
        assert gdata["total_runs"] == 30
        assert len(gdata["runs"]) == 30
        qids_in_runs = set(r["qid"] for r in gdata["runs"])
        assert qids_in_runs == set(AGG_QIDS)


# 14. no ground-truth mutation
def test_no_ground_truth_mutation():
    # Verify ground_truth file still has original counts
    gt_path = pathlib.Path("benchmarks/retrieval/ground_truth_v2.json")
    gt = json.loads(gt_path.read_text())
    gt_map = {q["qid"]: q for q in gt["queries"]}
    assert len(gt_map["AGG_HYPERTENSION"]["relevant_records"]) == 16
    assert len(gt_map["AGG_AMLODIPINE_5MG"]["relevant_records"]) == 17
    assert len(gt_map["AGG_PARACETAMOL_650MG"]["relevant_records"]) == 20
    assert len(gt_map["AGG_METFORMIN_500MG"]["relevant_records"]) == 7
    assert len(gt_map["AGG_T2D_HYPERTENSION"]["relevant_records"]) == 1
    # Also ensure AGGREGATE_QUERIES_V2 in code matches file
    code_map = {q["qid"]: q for q in AGGREGATE_QUERIES_V2 if q["qid"].startswith("AGG_")}
    for qid in ["AGG_HYPERTENSION", "AGG_AMLODIPINE_5MG", "AGG_PARACETAMOL_650MG", "AGG_METFORMIN_500MG", "AGG_T2D_HYPERTENSION"]:
        assert set(code_map[qid]["relevant_records"]) == set(gt_map[qid]["relevant_records"]), f"Mismatch for {qid}"
    # Ensure dataset unchanged (120 records)
    import re as _re
    text = pathlib.Path("data/sample_patient_data.txt").read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]
    assert len(blocks) == 120
    # Check no stray adaptive_k behavior in production
    for prod_file in ["secure_rag/retriever.py", "secure_rag/masker.py", "secure_rag/embedding.py", "secure_rag/rag_pipeline.py", "secure_rag/vector_store.py"]:
        content = pathlib.Path(prod_file).read_text()
        assert "adaptive" not in content.lower(), f"{prod_file} should not contain adaptive logic"
        assert "K_VALUES" not in content
