"""Tests for v2 masked-record generation metrics and Ollama instrumentation fixes.

Covers:
 1. record-level generation metric (masked)
 2. masked context not requiring raw MRNs
 3. Ollama done_reason extraction
 4. Ollama eval_count extraction
 5. truncation logic
 6. unknown termination handling
 7. estimated-vs-provider token accounting
"""

import json
import re
import pathlib
from unittest.mock import patch, MagicMock

import pytest

from benchmarks.retrieval.generation.generation_metrics import (
    EXPECTED_HYPERTENSION_MRNS,
    build_masked_record_fingerprints,
    compute_masked_generation_metrics,
)
from secure_rag.generator import get_last_ollama_metadata


# Helper: build minimal masked chunks for testing
def _masked_chunk(age, gender, admission, diagnosis, treatment):
    # Simplified masked chunk mimicking production format
    return (
        f"[NAME_MASKED], a {age}-year-old {gender} patient, was admitted to [ORG_MASKED] on [DOB_MASKED] with {admission}. "
        f"Medical ID: [PATIENT_ID_MASKED], Diagnosis: {diagnosis}. Treatment: {treatment}. "
        f"Notes: On examination, blood pressure was 146/90 mmHg. Follow-up: Review after 2 weeks."
    )


# 1. Record-level generation metric (masked) – deterministic
def test_masked_metric_deterministic():
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
    gm1 = compute_masked_generation_metrics(answer, chunks, ids, ["MRN1107", "MRN1066", "MRN1021"])
    gm2 = compute_masked_generation_metrics(answer, chunks, ids, ["MRN1107", "MRN1066", "MRN1021"])
    assert gm1 == gm2
    assert gm1["relevant_records_found"] == 2
    assert gm1["generation_recall"] == pytest.approx(2 / 3)


def test_masked_metric_relevant_coverage():
    chunks = [
        _masked_chunk(53, "female", "persistent hypertension", "Hypertension", "Telmisartan 40mg, Low-salt diet"),
        _masked_chunk(43, "female", "elevated blood pressure readings", "Hypertension", "Telmisartan 40mg, Low-salt diet"),
        _masked_chunk(52, "female", "high blood pressure; Fibromyalgia", "Hypertension; Fibromyalgia", "Amlodipine 5mg, Lifestyle modification"),
    ]
    ids = ["MRN1107", "MRN1066", "MRN1026"]
    answer = "1. [NAME_MASKED], a 53-year-old female patient, was admitted to [ORG_MASKED] on [DOB_MASKED] with persistent hypertension.\n2. [NAME_MASKED], a 52-year-old female patient, was admitted to [ORG_MASKED] on [DOB_MASKED] with high blood pressure; Fibromyalgia."
    gm = compute_masked_generation_metrics(answer, chunks, ids, ["MRN1107", "MRN1066", "MRN1026"])
    assert gm["relevant_records_found"] == 2
    assert "MRN1066" in gm["missing_records"]
    assert gm["missing_count"] == 1


def test_masked_metric_duplicate_detection():
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


def test_masked_metric_unsupported_detection():
    # Include a non-relevant CKD record
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
    # MRN1054 is non-relevant but in retrieved – should be unsupported
    assert gm["unsupported_count"] >= 1
    assert "MRN1054" in gm["unsupported_records"]


# 2. Masked context not requiring raw MRNs
def test_masked_context_no_raw_mrn_required():
    chunks = [
        _masked_chunk(53, "female", "persistent hypertension", "Hypertension", "Telmisartan 40mg"),
        _masked_chunk(43, "female", "elevated blood pressure readings", "Hypertension", "Amlodipine 5mg"),
    ]
    ids = ["MRN1107", "MRN1066"]
    # Answer contains NO raw MRNs, only masked placeholders
    answer = (
        "Patients with hypertension:\n"
        "1. [NAME_MASKED], a 53-year-old female patient, was admitted to [ORG_MASKED] on [DOB_MASKED] with persistent hypertension. Medical ID: [PATIENT_ID_MASKED]\n"
        "2. [NAME_MASKED], a 43-year-old female patient, was admitted to [ORG_MASKED] on [DOB_MASKED] with elevated blood pressure readings."
    )
    assert "MRN1107" not in answer
    assert "[PATIENT_ID_MASKED]" in answer
    gm = compute_masked_generation_metrics(answer, chunks, ids, ["MRN1107", "MRN1066"])
    assert gm["relevant_records_found"] == 2
    assert gm["generation_recall"] == pytest.approx(1.0)


def test_raw_mrn_in_context_invalid():
    # Ensure context audit detects raw MRNs
    from benchmarks.retrieval.generation.generation_budget_runner import _verify_frozen_context

    raw_context = "Patient MRN1001 with hypertension. Medical ID: MRN1001"
    masked_context = _masked_chunk(53, "female", "persistent hypertension", "Hypertension", "Telmisartan")
    audit_raw = _verify_frozen_context(raw_context, [raw_context])
    audit_masked = _verify_frozen_context(masked_context, [masked_context])
    assert audit_raw["context_contains_raw"] is True
    assert audit_masked["context_contains_raw"] is False
    assert audit_masked["has_masked_placeholder"] is True


# 3. Ollama done_reason extraction
def test_ollama_done_reason_extraction():
    from secure_rag import generator
    import json as j
    from urllib import request

    # Simulate Ollama streaming: two chunks, last has done=True and done_reason
    fake_chunks = [
        j.dumps({"response": "hello ", "done": False}).encode() + b"\n",
        j.dumps({"response": "world", "done": False}).encode() + b"\n",
        j.dumps({"done": True, "done_reason": "length", "eval_count": 200, "prompt_eval_count": 50}).encode() + b"\n",
    ]

    class FakeResponse:
        def __init__(self, lines):
            self.lines = lines
        def __enter__(self):
            return self.lines
        def __exit__(self, *args):
            return False

    with patch.object(request, "urlopen", return_value=FakeResponse(fake_chunks)):
        # Need to set env
        with patch.object(generator, "LLM_PROVIDER", "ollama"):
            # Also need to patch os.getenv for model
            result = list(generator._generate_ollama("ctx", "q", num_predict=200))
            assert result == ["hello ", "world"]
            meta = generator.get_last_ollama_metadata()
            assert meta["done"] is True
            assert meta["done_reason"] == "length"
            assert meta["eval_count"] == 200
            assert meta["prompt_eval_count"] == 50


def test_ollama_done_reason_stop():
    from secure_rag import generator
    import json as j
    from urllib import request

    fake_chunks = [
        j.dumps({"response": "answer", "done": False}).encode() + b"\n",
        j.dumps({"done": True, "done_reason": "stop", "eval_count": 6}).encode() + b"\n",
    ]

    class FakeResponse:
        def __init__(self, lines):
            self.lines = lines
        def __enter__(self):
            return self.lines
        def __exit__(self, *args):
            return False

    with patch.object(request, "urlopen", return_value=FakeResponse(fake_chunks)):
        with patch.object(generator, "LLM_PROVIDER", "ollama"):
            result = list(generator._generate_ollama("ctx", "q"))
            assert result == ["answer"]
            meta = generator.get_last_ollama_metadata()
            assert meta["done_reason"] == "stop"
            assert meta["eval_count"] == 6


# 4. Ollama eval_count extraction
def test_ollama_eval_count_extraction():
    from secure_rag import generator
    import json as j
    from urllib import request

    fake_chunks = [
        j.dumps({"response": "x", "done": False}).encode() + b"\n",
        j.dumps({"done": True, "done_reason": "stop", "eval_count": 42, "prompt_eval_count": 100, "total_duration": 12345}).encode() + b"\n",
    ]

    class FakeResponse:
        def __init__(self, lines):
            self.lines = lines
        def __enter__(self):
            return self.lines
        def __exit__(self, *args):
            return False

    with patch.object(request, "urlopen", return_value=FakeResponse(fake_chunks)):
        with patch.object(generator, "LLM_PROVIDER", "ollama"):
            list(generator._generate_ollama("ctx", "q"))
            meta = generator.get_last_ollama_metadata()
            assert meta["eval_count"] == 42
            assert meta["prompt_eval_count"] == 100
            assert meta["total_duration"] == 12345


# 5. Truncation logic
def test_truncation_logic_length():
    from benchmarks.retrieval.generation.generation_budget_runner import _call_generation
    from secure_rag import generator

    # Mock generate_answer to produce a short answer and set metadata with done_reason length
    fake_meta = {"done": True, "done_reason": "length", "eval_count": 200, "prompt_eval_count": 1000}

    with patch("benchmarks.retrieval.generation.generation_budget_runner.generate_answer", return_value=iter(["truncated answer"])):
        with patch("benchmarks.retrieval.generation.generation_budget_runner.get_last_ollama_metadata", return_value=fake_meta):
            gen = _call_generation("ctx", "q", budget=200)
            assert gen["finish_reason"] == "length"
            assert gen["truncation"] is True
            assert gen["truncation_status"] == "truncated"
            assert gen["output_tokens"] == 200
            assert gen["output_tokens_is_estimated"] is False


def test_truncation_logic_stop():
    from benchmarks.retrieval.generation.generation_budget_runner import _call_generation

    fake_meta = {"done": True, "done_reason": "stop", "eval_count": 150, "prompt_eval_count": 1000}

    with patch("benchmarks.retrieval.generation.generation_budget_runner.generate_answer", return_value=iter(["full answer"])):
        with patch("benchmarks.retrieval.generation.generation_budget_runner.get_last_ollama_metadata", return_value=fake_meta):
            gen = _call_generation("ctx", "q", budget=1200)
            assert gen["finish_reason"] == "stop"
            assert gen["truncation"] is False
            assert gen["truncation_status"] == "not_truncated"
            assert gen["output_tokens"] == 150
            assert gen["output_tokens_is_estimated"] is False


# 6. Unknown termination handling
def test_unknown_termination_when_missing():
    from benchmarks.retrieval.generation.generation_budget_runner import _call_generation

    fake_meta = {}  # no done_reason, no eval_count
    with patch("benchmarks.retrieval.generation.generation_budget_runner.generate_answer", return_value=iter(["answer"])):
        with patch("benchmarks.retrieval.generation.generation_budget_runner.get_last_ollama_metadata", return_value=fake_meta):
            gen = _call_generation("ctx", "q", budget=200)
            assert gen["finish_reason"] == "unknown"
            assert gen["truncation"] is None
            assert gen["truncation_status"] == "unknown"
            assert gen["output_tokens_is_estimated"] is True


def test_unknown_truncation_not_false():
    # When done_reason missing, truncation must be None/unknown, not False
    from benchmarks.retrieval.generation.generation_metrics import aggregate_by_budget

    results = [
        {"generation_budget": 200, "generation_recall": 0.5, "missing_count": 8, "duplicate_records": 0, "unsupported_count": 0, "truncation": None, "latency_ms": 100, "output_tokens": 150, "truncation_status": "unknown"},
        {"generation_budget": 200, "generation_recall": 0.6, "missing_count": 6, "duplicate_records": 0, "unsupported_count": 0, "truncation": None, "latency_ms": 110, "output_tokens": 160, "truncation_status": "unknown"},
    ]
    agg = aggregate_by_budget(results)
    assert agg["200"]["truncation_count"] == 0
    assert agg["200"]["unknown_termination_count"] == 2
    assert agg["200"]["unknown_termination_rate"] == pytest.approx(1.0)
    assert agg["200"]["truncation_rate"] == pytest.approx(0.0)


# 7. Estimated vs provider token accounting
def test_token_accounting_provider_vs_estimated():
    from benchmarks.retrieval.generation.generation_budget_runner import _call_generation

    # Provider provides eval_count -> not estimated
    fake_meta = {"done": True, "done_reason": "stop", "eval_count": 123, "prompt_eval_count": 456}
    with patch("benchmarks.retrieval.generation.generation_budget_runner.generate_answer", return_value=iter(["ans"])):
        with patch("benchmarks.retrieval.generation.generation_budget_runner.get_last_ollama_metadata", return_value=fake_meta):
            gen = _call_generation("ctx", "q", budget=1200)
            assert gen["output_tokens"] == 123
            assert gen["output_tokens_is_estimated"] is False
            assert gen["prompt_tokens"] == 456
            assert gen["prompt_tokens_is_estimated"] is False

    # Provider missing eval_count -> estimated
    fake_meta2 = {}
    with patch("benchmarks.retrieval.generation.generation_budget_runner.generate_answer", return_value=iter(["ans with some words"])):
        with patch("benchmarks.retrieval.generation.generation_budget_runner.get_last_ollama_metadata", return_value=fake_meta2):
            gen2 = _call_generation("ctx", "q", budget=200)
            assert gen2["output_tokens_is_estimated"] is True
            assert gen2["prompt_tokens_is_estimated"] is True
            assert gen2["output_tokens"] is not None
