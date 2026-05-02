import numpy as np

from secure_rag import build_rag
from secure_rag.rag_pipeline import clean_input_text


def test_build_rag(tmp_path, monkeypatch):
    file_path = tmp_path / "test_data.txt"
    file_path.write_text("RAG stands for Retrieval Augmented Generation.", encoding="utf-8")

    monkeypatch.setattr(
        "secure_rag.rag_pipeline.embed_chunks",
        lambda chunks: np.zeros((len(chunks), 4), dtype="float32"),
    )

    vs, chunks = build_rag(str(file_path))

    assert vs is not None
    assert len(chunks) > 0


def test_clean_input_text_removes_prompt_artifacts():
    dirty_text = """Secure RAG Chat
Type 'exit' to quit

Priya Patel was admitted to Fortis Hospital after experiencing chest pain.

Context:
[NAME_MASKED] was admitted to [ORG_MASKED].

Question:
how old is priya patel? [/INST]

LLM:
You: exit
"""

    cleaned = clean_input_text(dirty_text)

    assert cleaned == "Priya Patel was admitted to Fortis Hospital after experiencing chest pain."


def test_build_rag_chunks_records_independently(tmp_path, monkeypatch):
    file_path = tmp_path / "records.txt"
    file_path.write_text(
        "Arjun Sharma has persistent fever.\n\nPriya Patel has chest pain.",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "secure_rag.rag_pipeline.embed_chunks",
        lambda chunks: np.zeros((len(chunks), 4), dtype="float32"),
    )

    _, chunks = build_rag(str(file_path))

    assert len(chunks) == 2
    assert "persistent fever" in chunks[0]
    assert "chest pain" in chunks[1]
