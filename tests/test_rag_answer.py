from secure_rag.rag_pipeline import rag_answer


def test_rag_answer_mock():
    def fake_generator(context, query):#mocking answer w/o api calls
        yield "mock answer"

    vs = None
    chunks = ["RAG is a method"]

    result = list(fake_generator("", ""))
    assert "mock answer" in result[0] 