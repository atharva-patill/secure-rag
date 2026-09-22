"""Research-only unit tests for generation-filtering experiment.

Covers 10 required checks:
1. filtering instruction exists
2. prompt contains filtering instruction
3. raw MRNs are not inserted into prompt
4. K remains 20
5. evaluator counts relevant records correctly
6. unsupported records are counted
7. duplicate records are counted
8. done_reason="length" → truncated=True
9. done_reason="stop" → truncated=False
10. missing done_reason → truncated=None
"""

import re
import pytest
from unittest.mock import patch

from benchmarks.retrieval.generation.generation_filter_metrics import FILTER_INSTRUCTION
from benchmarks.retrieval.generation.generation_filter_runner import (
    K,
    FILTER_INSTRUCTION as RUNNER_FILTER,
    _build_filtered_ollama_prompt,
    _verify_prompt_no_raw_mrns,
    _context_hash,
)
from benchmarks.retrieval.generation.generation_metrics import (
    EXPECTED_K,
    EXPECTED_HYPERTENSION_COUNT,
    compute_masked_generation_metrics,
    build_masked_record_fingerprints,
)


def _masked_chunk(age, gender, admission, diagnosis, treatment):
    return (
        f"[NAME_MASKED], a {age}-year-old {gender} patient, was admitted to [ORG_MASKED] on [DOB_MASKED] with {admission}. "
        f"Medical ID: [PATIENT_ID_MASKED], Diagnosis: {diagnosis}. Treatment: {treatment}. "
        f"Notes: On examination, blood pressure was 146/90 mmHg. Follow-up: Review after 2 weeks."
    )


# 1. filtering instruction exists
def test_filter_instruction_exists():
    assert FILTER_INSTRUCTION is not None
    assert isinstance(FILTER_INSTRUCTION, str)
    assert len(FILTER_INSTRUCTION) > 20
    assert "Enumerate only records whose Diagnosis contains Hypertension" in FILTER_INSTRUCTION
    assert "Do not enumerate retrieved records whose Diagnosis does not contain Hypertension" in FILTER_INSTRUCTION
    assert "Do not infer, combine, or invent patient records" in FILTER_INSTRUCTION
    assert "Each enumerated patient must correspond to a retrieved record whose Diagnosis explicitly contains Hypertension" in FILTER_INSTRUCTION
    # Runner and metrics must agree
    assert RUNNER_FILTER == FILTER_INSTRUCTION


# 2. prompt contains filtering instruction
def test_prompt_contains_filter_instruction():
    ctx = _masked_chunk(53, "female", "persistent hypertension", "Hypertension", "Telmisartan 40mg")
    prompt = _build_filtered_ollama_prompt(ctx, "Give me all patients with hypertension.")
    assert FILTER_INSTRUCTION in prompt
    audit = _verify_prompt_no_raw_mrns(prompt)
    assert audit["has_filter_instruction"] is True


# 3. raw MRNs are not inserted into prompt
def test_raw_mrns_not_in_prompt():
    ctx = "\n\n".join([
        _masked_chunk(53, "female", "persistent hypertension", "Hypertension", "Telmisartan 40mg"),
        _masked_chunk(43, "female", "elevated blood pressure readings", "Hypertension", "Amlodipine 5mg"),
    ])
    prompt = _build_filtered_ollama_prompt(ctx, "Give me all patients with hypertension.")
    # No raw MRN pattern
    assert re.search(r"\bMRN\d+\b", prompt) is None
    audit = _verify_prompt_no_raw_mrns(prompt)
    assert audit["prompt_contains_raw"] is False
    assert audit["raw_mrn_count_prompt"] == 0
    # Also ensure context hash does not imply prompt raw leak
    assert "[PATIENT_ID_MASKED]" in prompt


# 4. K remains 20
def test_k_remains_20():
    assert K == 20
    assert EXPECTED_K == 20
    # Ensure runner K matches expected
    assert K == EXPECTED_K


# 5. evaluator counts relevant records correctly
def test_evaluator_counts_relevant_correctly():
    chunks = [
        _masked_chunk(53, "female", "persistent hypertension", "Hypertension", "Telmisartan 40mg, Low-salt diet"),
        _masked_chunk(43, "female", "elevated blood pressure readings", "Hypertension", "Telmisartan 40mg, Low-salt diet"),
        _masked_chunk(66, "male", "persistent hypertension", "Hypertension", "Amlodipine 5mg, Lifestyle modification"),
    ]
    ids = ["MRN1107", "MRN1066", "MRN1021"]
    answer = (
        "1. [NAME_MASKED], a 53-year-old female patient, was admitted to [ORG_MASKED] on [DOB_MASKED] with persistent hypertension. Medical ID: [PATIENT_ID_MASKED], Diagnosis: Hypertension.\n"
        "2. [NAME_MASKED], a 43-year-old female patient, was admitted to [ORG_MASKED] on [DOB_MASKED] with elevated blood pressure readings."
    )
    gm = compute_masked_generation_metrics(answer, chunks, ids, ["MRN1107", "MRN1066", "MRN1021"])
    assert gm["relevant_records_found"] == 2
    assert gm["generation_recall"] == pytest.approx(2 / 3)
    assert gm["missing_count"] == 1
    assert "MRN1021" in gm["missing_records"]


# 6. unsupported records are counted
def test_evaluator_unsupported_counted():
    chunks = [
        _masked_chunk(53, "female", "persistent hypertension", "Hypertension", "Telmisartan 40mg"),
        _masked_chunk(47, "male", "renal dysfunction", "Chronic Kidney Disease", "Low-salt diet"),
    ]
    ids = ["MRN1107", "MRN1054"]
    answer = (
        "1. [NAME_MASKED], a 53-year-old female patient, was admitted to [ORG_MASKED] on [DOB_MASKED] with persistent hypertension.\n"
        "2. [NAME_MASKED], a 47-year-old male patient, was admitted to [ADDRESS_MASKED] on [DOB_MASKED] with renal dysfunction."
    )
    gm = compute_masked_generation_metrics(answer, chunks, ids, ["MRN1107"])
    assert gm["unsupported_count"] >= 1
    assert "MRN1054" in gm["unsupported_records"]
    # Also check hallucinated not conflated when supported
    assert gm["relevant_records_found"] == 1


# 7. duplicate records are counted
def test_evaluator_duplicate_counted():
    chunks = [
        _masked_chunk(53, "female", "persistent hypertension", "Hypertension", "Telmisartan 40mg, Low-salt diet"),
    ]
    ids = ["MRN1107"]
    answer = (
        "1. [NAME_MASKED], a 53-year-old female patient, was admitted to [ORG_MASKED] on [DOB_MASKED] with persistent hypertension.\n"
        "2. [NAME_MASKED], a 53-year-old female patient, was admitted to [ORG_MASKED] on [DOB_MASKED] with persistent hypertension."
    )
    gm = compute_masked_generation_metrics(answer, chunks, ids, ["MRN1107"])
    assert gm["duplicate_records"] == 1
    assert gm["duplicate_mrns"] == ["MRN1107"]


# 8. done_reason="length" → truncated=True
def test_truncation_length():
    from benchmarks.retrieval.generation.generation_filter_runner import _call_generation_filtered

    fake_meta = {"done": True, "done_reason": "length", "eval_count": 800, "prompt_eval_count": 3466}
    with patch("benchmarks.retrieval.generation.generation_filter_runner.generate_answer", return_value=iter(["ans"])):
        with patch("benchmarks.retrieval.generation.generation_filter_runner.get_last_ollama_metadata", return_value=fake_meta):
            gen = _call_generation_filtered("ctx", "Give me all patients with hypertension.", budget=800)
            assert gen["finish_reason"] == "length"
            assert gen["truncation"] is True
            assert gen["truncation_status"] == "truncated"
            assert gen["done_reason"] == "length"
            assert gen["output_tokens"] == 800
            assert gen["output_tokens_is_estimated"] is False


# 9. done_reason="stop" → truncated=False
def test_truncation_stop():
    from benchmarks.retrieval.generation.generation_filter_runner import _call_generation_filtered

    fake_meta = {"done": True, "done_reason": "stop", "eval_count": 753, "prompt_eval_count": 3466}
    with patch("benchmarks.retrieval.generation.generation_filter_runner.generate_answer", return_value=iter(["ans"])):
        with patch("benchmarks.retrieval.generation.generation_filter_runner.get_last_ollama_metadata", return_value=fake_meta):
            gen = _call_generation_filtered("ctx", "Give me all patients with hypertension.", budget=800)
            assert gen["finish_reason"] == "stop"
            assert gen["truncation"] is False
            assert gen["truncation_status"] == "not_truncated"
            assert gen["done_reason"] == "stop"
            assert gen["output_tokens"] == 753
            assert gen["output_tokens_is_estimated"] is False


# 10. missing done_reason → truncated=None
def test_truncation_missing():
    from benchmarks.retrieval.generation.generation_filter_runner import _call_generation_filtered

    fake_meta = {}  # no done_reason
    with patch("benchmarks.retrieval.generation.generation_filter_runner.generate_answer", return_value=iter(["ans"])):
        with patch("benchmarks.retrieval.generation.generation_filter_runner.get_last_ollama_metadata", return_value=fake_meta):
            gen = _call_generation_filtered("ctx", "Give me all patients with hypertension.", budget=800)
            assert gen["finish_reason"] == "unknown"
            assert gen["truncation"] is None
            assert gen["truncation_status"] == "unknown"
            assert gen["done_reason"] is None
            assert gen["output_tokens_is_estimated"] is True
            # Must not silently be False
            assert gen["truncation"] is not False

