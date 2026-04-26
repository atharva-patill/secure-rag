from secure_rag import build_rag
from secure_rag.rag_pipeline import clean_input_text


def test_build_rag():
    with open("test_data.txt", "w") as f:
        f.write("RAG stands for Retrieval Augmented Generation.")

    vs, chunks = build_rag("test_data.txt")

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
