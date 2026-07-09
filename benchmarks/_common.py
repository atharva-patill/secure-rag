"""
Shared utilities for Secure RAG benchmark evaluation.

Factored out of privacy_eval.py so both privacy and retrieval
evaluation can consume the same dataset, configuration registry,
and utility functions without duplication.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from secure_rag import rag_answer
from secure_rag.generator import generate_answer
from secure_rag.masker import mask_text
from secure_rag.pdf_loader import chunk_text
from secure_rag.embedding import embed_chunks
from secure_rag.retriever import retrieve
from secure_rag.vector_store import VectorStore

BENCHMARK_DIR = Path(__file__).parent
DATASET_PATH = BENCHMARK_DIR / "dataset.jsonl"
QUERIES_PATH = BENCHMARK_DIR / "dataset_queries.json"
SPLIT_PATH = BENCHMARK_DIR / "train_test_split.json"
RESULTS_PATH = BENCHMARK_DIR / "results.json"

PII_FIELDS = ["name", "aadhaar", "pan", "phone", "email", "mrn", "dob", "address"]
RETRIEVAL_K = 5

DEBUG = False
MAX_LLM_FAILURE_RATE = 0.30
LLM_RETRIES = 1


def normalize(text: str) -> str:
    return "".join(text.lower().split())


def loose_match(value: str, text: str) -> bool:
    pattern = re.sub(r"\s+", r"\\s*", re.escape(value))
    return re.search(pattern, text, re.IGNORECASE) is not None


def pii_leaks(text: str, pii: dict) -> bool:
    for value in pii.values():
        if not value:
            continue
        if normalize(value) in normalize(text):
            return True
        if loose_match(value, text):
            return True
    return False


def load_records() -> Dict[str, dict]:
    records = {}
    with open(DATASET_PATH) as f:
        for line in f:
            r = json.loads(line)
            records[r["record_id"]] = r
    return records


def load_queries() -> list:
    with open(QUERIES_PATH) as f:
        return json.load(f)


def load_split() -> dict:
    with open(SPLIT_PATH) as f:
        return json.load(f)


def build_index(records: dict, use_masking: bool) -> Tuple[VectorStore, List[str]]:
    texts = []
    for rid, r in records.items():
        text = r["text"]
        if use_masking:
            text = mask_text(text)
        texts.append(text)

    chunks = []
    for text in texts:
        chunks.extend(chunk_text(text))

    embeddings = embed_chunks(chunks)
    vector_store = VectorStore(embeddings)
    return vector_store, chunks


EVALUATION_CONFIGS = [
    {
        "id": "baseline_a",
        "label": "Baseline A",
        "display_name": "Baseline A \u2014 Raw Retrieval-Augmented Generation",
        "description": "No masking is applied during indexing, retrieval or answer generation.",
        "get_idx": lambda r_idx, r_ck, p_idx, p_ck: (r_idx, r_ck),
        "answer": lambda q, idx, ck: benchmark_answer(q, idx, ck, "baseline_a"),
    },
    {
        "id": "baseline_b",
        "label": "Baseline B",
        "display_name": "Baseline B \u2014 Post-Retrieval Privacy Masking",
        "description": "Documents are indexed without masking. Retrieved context is masked immediately before answer generation.",
        "get_idx": lambda r_idx, r_ck, p_idx, p_ck: (r_idx, r_ck),
        "answer": lambda q, idx, ck: benchmark_answer(q, idx, ck, "baseline_b"),
    },
    {
        "id": "secure_rag",
        "label": "Secure RAG",
        "display_name": "Proposed Method \u2014 Secure RAG",
        "description": "Sensitive entities are masked before chunking and embedding. The vector store never contains raw sensitive information. No answer-time masking is performed.",
        "get_idx": lambda r_idx, r_ck, p_idx, p_ck: (p_idx, p_ck),
        "answer": lambda q, idx, ck: benchmark_answer(q, idx, ck, "secure_rag"),
    },
]


def benchmark_answer(query: str, index, chunks: List[str], config_id: str) -> str:
    if config_id == "secure_rag":
        return "".join(list(rag_answer(query, index, chunks))).strip()

    context_chunks = retrieve(query, index, chunks)
    context = "\n\n".join(chunk for chunk in context_chunks if chunk)

    if config_id == "baseline_b":
        context = mask_text(context)
    elif config_id != "baseline_a":
        raise ValueError(f"Unsupported benchmark config: {config_id}")

    response = "".join(generate_answer(context, f"{query}\n\nAnswer:"))
    return truncate_at_stop_marker(response)


def truncate_at_stop_marker(text: str) -> str:
    stop_markers = ("\nContext:", "\nQuestion:", "[/INST]")
    positions = [text.find(m) for m in stop_markers if m in text]
    if positions:
        text = text[: min(positions)]
    return text.strip()
