# AGENTS.md

## Commands

- Run tests: `python3 -m pytest tests/`
- Skip spaCy-dependent tests: `python3 -m pytest tests/ -m "not slow"`
- Run one test: `python3 -m pytest tests/test_masker.py::TestNERMasking::test_person_name_masked`
- Run privacy evaluation: `python3 benchmarks/privacy_eval.py`
- Regenerate benchmark dataset: `python3 benchmarks/generate_dataset.py`
- Build package: `python3 -m build`

## Setup Gotchas

- Local NER tests require the spaCy model `en_core_web_sm`. Install it with `pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl --break-system-packages` if `python3 -m spacy download ...` fails.
- `HF_API_KEY` is required for LLM-backed answer generation and the `PHI in Answers` part of `benchmarks/privacy_eval.py`.
- `HF_TOKEN` may be needed for embedding model download in `secure_rag/embedding.py`.
- Dev extras are defined in `pyproject.toml`: install with `pip3 install -e ".[dev]" --break-system-packages` if needed.

## Package Boundaries

- `secure_rag/` is the library/package shipped by `pyproject.toml`.
- `benchmarks/` is intentionally separate from `secure_rag/` so evaluation code and synthetic datasets do not bloat the package.
- Main library entrypoints are `secure_rag/rag_pipeline.py`:
  - `build_rag(file_path, use_masking=True)` controls pre-embedding masking.
  - `rag_answer(query, vector_store, chunks, mask_mode="raw")` controls inference-time mode.

## Privacy Modes

- `raw`: build with `use_masking=False`, answer with `mask_mode="raw"`.
- `post`: build with `use_masking=False`, answer with `mask_mode="post"`.
- `pre`: build with `use_masking=True`, answer with `mask_mode="pre"`.

Critical rules verified in code:

- Never mask the query in any mode.
- `pre` masking happens only inside `build_rag()` before chunking/embedding.
- `post` masks only retrieved context inside `rag_answer()` before generation.
- `raw` and `post` must share the same raw index. Do not create a separate post-masked index.

## Evaluation Expectations

- `benchmarks/privacy_eval.py` compares exactly three modes: `raw`, `post`, `pre`.
- It builds exactly two indices: raw and pre-masked.
- Current reported metrics are:
  - Document Leakage
  - Retrieval Leakage (`k=5`)
  - Masking Recall
  - PHI in Answers
- Leakage checks use normalization and regex-based loose matching; do not revert to naive substring checks.

## Testing Notes

- `tests/test_masker.py` slow tests depend on spaCy NER.
- CI installs the spaCy model before running `pytest`; local runs may fail if the model is missing.
- If changing masking or pipeline behavior, run both:
  - `python3 -m pytest tests/`
  - `python3 benchmarks/privacy_eval.py`
