"""Focused tests for query-routed fixed-depth retrieval.

Implements sections 8A-8H from the spec; 8I is covered by running all tests.
"""

import re
import inspect

import numpy as np
import pytest

from secure_rag.query_router import (
    AGGREGATE_K,
    DEFAULT_SINGLE_K,
    classify_query,
    routed_k,
)
from secure_rag.rag_pipeline import rag_answer
from secure_rag.retriever import retrieve
from secure_rag.vector_store import VectorStore


# ---- A. Default behavior: without routed mode still K=2 ----

def test_default_behavior_uses_k2(monkeypatch):
    captured = {}

    def fake_retrieve(query, vector_store, chunks, k=2):
        captured["k"] = k
        return ["chunk one", "chunk two"]

    def fake_generate(context, query):
        yield "ok"

    monkeypatch.setattr("secure_rag.rag_pipeline.retrieve", fake_retrieve)
    monkeypatch.setattr("secure_rag.rag_pipeline.generate_answer", fake_generate)

    result = list(rag_answer("Which hospital did the patient visit?", object(), ["unused"]))
    assert captured["k"] == 2
    assert result == ["ok"]

    # explicit routed=False also 2
    captured.clear()
    result = list(rag_answer("Which patients were prescribed Amlodipine 5mg?", object(), ["unused"], routed=False))
    assert captured["k"] == 2

    # signature preserves backward compat: call without routed kwarg
    sig = inspect.signature(rag_answer)
    assert "routed" in sig.parameters
    assert sig.parameters["routed"].default is False
    # existing callers with 3 args still work
    captured.clear()
    result = list(rag_answer("hello", object(), ["unused"]))
    assert captured["k"] == 2


# ---- B. Single classification ----

@pytest.mark.parametrize("query", [
    "Which hospital did the patient visit?",
    "What is the patient's phone number?",
    "What is the patient's Aadhaar number?",
    "What is the MRN of the patient?",
    "Summarize the patient record.",
    "How old is the patient?",
])
def test_single_classification(query):
    assert classify_query(query) == "single", f"Expected single for {query!r}"


# ---- C. Aggregate / multi classification (uses repository vocabulary) ----

@pytest.mark.parametrize("query", [
    "Which patients were prescribed Amlodipine 5mg?",
    "Which patients have Hypertension?",
    "Who received Paracetamol 650mg?",
    "Which patients are being treated for both Type 2 Diabetes and Hypertension?",
    "Who all have hypertension?",
    "List all patients with Hypertension",
    "How many patients were prescribed Metformin 500mg?",
    "Which records have Hypertension?",
    "Summarize all patients with Hypertension",
])
def test_multi_classification(query):
    assert classify_query(query) == "multi", f"Expected multi for {query!r}"


# ---- D. Routing: single->2 multi->20 ----

def test_routing_single_to_2():
    assert routed_k("Which hospital did the patient visit?", enabled=True) == 2
    assert routed_k("Summarize the patient record.", enabled=True) == 2

def test_routing_multi_to_20():
    assert routed_k("Which patients were prescribed Amlodipine 5mg?", enabled=True) == 20
    assert routed_k("Who received Paracetamol 650mg?", enabled=True) == 20
    assert routed_k("List all patients with Hypertension", enabled=True) == 20
    assert routed_k("How many patients have Hypertension?", enabled=True) == 20

def test_routing_disabled_always_2():
    for q in ["Which patients have Hypertension?", "Which hospital did the patient visit?"]:
        assert routed_k(q, enabled=False) == 2

def test_constants_centralized():
    assert DEFAULT_SINGLE_K == 2
    assert AGGREGATE_K == 20


# ---- E. Retrieval count: 2 vs 20 with enough chunks ----

def _make_store(n_chunks, dim=4):
    embeddings = np.zeros((n_chunks, dim), dtype="float32")
    # make embeddings distinct so search order deterministic: ramp
    for i in range(n_chunks):
        embeddings[i, 0] = float(i)
    return VectorStore(embeddings), [f"chunk {i}" for i in range(n_chunks)]


def test_retrieval_count_single_vs_multi(monkeypatch):
    # monkeypatch embed_chunks to avoid model download
    monkeypatch.setattr("secure_rag.retriever.embed_chunks", lambda chunks: np.zeros((len(chunks), 4), dtype="float32"))
    vs, chunks = _make_store(30)
    # via rag_answer routed flag
    captured = {}
    orig_retrieve = retrieve

    def fake_generate(context, query):
        # count chunks in context
        captured["ctx_chunks"] = context.split("\n\n") if context else []
        yield "done"

    monkeypatch.setattr("secure_rag.rag_pipeline.generate_answer", fake_generate)

    # single -> 2
    cap_k = {}
    def capture_retrieve(query, vector_store, chunks_arg, k=2):
        cap_k["k"] = k
        return orig_retrieve(query, vector_store, chunks_arg, k=k)

    monkeypatch.setattr("secure_rag.rag_pipeline.retrieve", capture_retrieve)
    list(rag_answer("Which hospital did the patient visit?", vs, chunks, routed=True))
    assert cap_k["k"] == 2
    # check context has 2 chunks (via retriever directly)
    res = retrieve("test query", vs, chunks, k=2)
    assert len(res) == 2

    list(rag_answer("Which patients have Hypertension?", vs, chunks, routed=True))
    assert cap_k["k"] == 20
    res = retrieve("test query", vs, chunks, k=20)
    assert len(res) == 20


# ---- F. Small index: min(20, ntotal) ----

def test_small_index_capped(monkeypatch):
    monkeypatch.setattr("secure_rag.retriever.embed_chunks", lambda chunks: np.zeros((len(chunks), 4), dtype="float32"))
    vs, chunks = _make_store(5)
    res = retrieve("Which patients have Hypertension?", vs, chunks, k=20)
    assert len(res) == 5
    # via routed path also capped
    captured = {}
    def fake_generate(context, query):
        captured["context"] = context
        yield "ok"
    monkeypatch.setattr("secure_rag.rag_pipeline.retrieve", lambda q, vs_, ch, k=2: retrieve(q, vs_, ch, k=k))
    monkeypatch.setattr("secure_rag.rag_pipeline.generate_answer", fake_generate)
    list(rag_answer("Which patients have Hypertension?", vs, chunks, routed=True))
    # context should have 5 chunks joined
    assert captured["context"].count("chunk") == 5


# ---- G. Privacy: routed still operates on already masked context ----

def test_routed_privacy_no_raw_mrn(monkeypatch):
    monkeypatch.setattr("secure_rag.retriever.embed_chunks", lambda chunks: np.zeros((len(chunks), 4), dtype="float32"))
    # chunks are already masked (contain [PATIENT_ID_MASKED] not MRN)
    masked_chunks = [
        "[NAME_MASKED] admitted to [ORG_MASKED] with Hypertension. Medical ID: [PATIENT_ID_MASKED].",
        "[NAME_MASKED] prescribed Amlodipine 5mg. Medical ID: [PATIENT_ID_MASKED].",
        "[NAME_MASKED] has Type 2 Diabetes. Medical ID: [PATIENT_ID_MASKED].",
    ] * 7  # 21 chunks
    vs, _ = _make_store(len(masked_chunks))
    # override embed for vs creation already done, but retrieval uses vs with 21 entries
    # Use actual chunks for context
    # Build vs with masked chunk embeddings (zeros)
    vs2 = VectorStore(np.zeros((len(masked_chunks), 4), dtype="float32"))

    captured = {}
    def fake_generate(context, query):
        captured["context"] = context
        yield "answer"

    monkeypatch.setattr("secure_rag.rag_pipeline.generate_answer", fake_generate)
    # single routed
    list(rag_answer("Which hospital did the patient visit?", vs2, masked_chunks, routed=True))
    assert "MRN" not in captured["context"]
    assert "[PATIENT_ID_MASKED]" in captured["context"]
    # multi routed (20 chunks)
    list(rag_answer("Which patients have Hypertension?", vs2, masked_chunks, routed=True))
    assert "MRN" not in captured["context"]
    # ensure no MRN pattern
    assert not re.search(r"\bMRN\d+\b", captured["context"])


# ---- H. Determinism ----

def test_determinism():
    queries = [
        "Which hospital did the patient visit?",
        "Which patients were prescribed Amlodipine 5mg?",
        "List all patients with Hypertension",
        "How many patients have Hypertension?",
    ]
    for q in queries:
        first = classify_query(q)
        second = classify_query(q)
        assert first == second
        assert routed_k(q, enabled=True) == routed_k(q, enabled=True)
        assert routed_k(q, enabled=False) == routed_k(q, enabled=False)


# ---- I. Regression: natural aggregate paraphrases (live validation) ----

@pytest.mark.parametrize("query", [
    "get everyone diagnosed with hypertension",
    "Show all patients taking amlodipine 5mg.",
    "Find every record containing paracetamol 650mg.",
    "Which records mention metformin 500mg?",
    "Give me everyone who has hypertension",
])
def test_regression_validation_aggregate_queries(query):
    assert classify_query(query) == "multi", f"Expected multi for {query!r}"
    assert routed_k(query, enabled=True) == AGGREGATE_K == 20
    assert routed_k(query, enabled=True) == 20
    # disabled / default remains K2
    assert routed_k(query, enabled=False) == DEFAULT_SINGLE_K == 2


# ---- J. Additional domain-independent aggregate paraphrases ----

@pytest.mark.parametrize("query", [
    "show every patient with diabetes",
    "list all records mentioning aspirin",
    "which patients received treatment X",
])
def test_domain_independent_aggregate_paraphrases(query):
    assert classify_query(query) == "multi", f"Expected multi for {query!r}"
    assert routed_k(query, enabled=True) == AGGREGATE_K == 20
