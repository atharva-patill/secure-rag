# Secure RAG Benchmark Dataset

Synthetic Indian medical dataset for evaluating privacy-aware RAG systems.

## Overview

- **120 medical records** (100 train / 20 test)
- **600+ queries** (4-5 per record)
- **Realistic Indian PHI** (Aadhaar, PAN, MRN, Phone, Email, DOB, Address)
- **Reproducible** (seed=42)

## Structure

### `dataset.jsonl`
One JSON object per line:
```json
{"record_id": "MED001", "text": "...", "pii": {"name": "...", "aadhaar": "...", ...}, "hospital": "..."}
```

### `dataset_queries.json`
```json
{"record_id": "MED001", "queries": [{"qid": "MED001_Q1", "question": "...", "field": "..."}]}
```

### `train_test_split.json`
```json
{"train": ["MED001", ...], "test": ["MED064", ...]}
```

## PHI Fields

| Field | Format | Example |
|-------|--------|---------|
| name | Indian full name | Rajesh Kumar |
| phone | 10-digit | 9876543210 |
| aadhaar | XXXX XXXX XXXX | 1234 5678 9012 |
| pan | ABCDE1234F | ABCDE1234F |
| email | email | name@example.com |
| mrn | MRN##### | MRN12345 |
| dob | DD/MM/YYYY | 15/08/1990 |
| address | Indian address | H.No. 123, MG Road, Mumbai |

## Usage

```python
import json

with open("benchmarks/dataset.jsonl") as f:
    for line in f:
        record = json.loads(line)
        print(record["record_id"], record["pii"]["mrn"])
```

## Regenerate

```bash
python3 benchmarks/generate_dataset.py
```

---

## Evaluation Configurations

The benchmark compares three privacy strategies:

### Baseline A — Raw Retrieval-Augmented Generation

No masking is applied during indexing, retrieval or answer generation.
Measures the baseline privacy leakage of standard RAG.

### Baseline B — Post-Retrieval Privacy Masking

Documents are indexed without masking.
Retrieved context is masked immediately before answer generation.
Isolates the privacy benefit of masking at inference time only.

### Proposed Method — Secure RAG

Sensitive entities are masked before chunking and embedding.
The vector store never contains raw sensitive information.
No answer-time masking is performed.
Represents the canonical Secure RAG pipeline.

### Metrics

- **Document Leakage:** Percentage of synthetic PII values present in all indexed chunks.
- **Retrieval Leakage (k=5):** Percentage of PII found in the top-5 retrieved chunks for each query.
- **Masking Recall:** Percentage of PII values successfully removed by the masker.
- **PHI in Answers (LLM):** Percentage of generated answers that contain raw PII.
  Requires `HF_API_KEY`.

### Usage

```bash
# Full evaluation (indexing + retrieval metrics + LLM)
HF_API_KEY=your_key python3 benchmarks/privacy_eval.py

# Indexing and retrieval metrics only (no LLM)
python3 benchmarks/privacy_eval.py
```
