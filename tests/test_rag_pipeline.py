from secure_rag import build_rag


def test_build_rag():
    with open("test_data.txt", "w") as f:
        f.write("RAG stands for Retrieval Augmented Generation.")

    vs, chunks = build_rag("test_data.txt")

    assert vs is not None
    assert len(chunks) > 0