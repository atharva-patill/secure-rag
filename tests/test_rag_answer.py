from secure_rag.rag_pipeline import rag_answer
from secure_rag.retriever import retrieve


def test_rag_answer_mock():
    def fake_generator(context, query):#mocking answer w/o api calls
        yield "mock answer"

    vs = None
    chunks = ["RAG is a method"]

    result = list(fake_generator("", ""))
    assert "mock answer" in result[0] 


def test_rag_answer_builds_single_prompt(monkeypatch):
    captured = {}

    def fake_retrieve(query, vector_store, chunks, k=2):
        return ["chunk one", "chunk two"]

    def fake_generate_answer(context, query):
        captured["context"] = context
        captured["query"] = query
        yield "I don't know"

    monkeypatch.setattr("secure_rag.rag_pipeline.retrieve", fake_retrieve)
    monkeypatch.setattr("secure_rag.rag_pipeline.generate_answer", fake_generate_answer)

    result = list(rag_answer("how old is arjun sharma?", object(), ["unused"]))

    assert result == ["I don't know"]
    assert captured["context"] == "chunk one\n\nchunk two"
    assert captured["query"] == "how old is arjun sharma?\n\nAnswer:"


def test_rag_answer_truncates_prompt_echo(monkeypatch):
    def fake_retrieve(query, vector_store, chunks, k=2):
        return ["chunk one"]

    def fake_generate_answer(context, query):
        yield "[ORG_MASKED]\n\nContext:\nchunk one\n\nQuestion:\nwhere was priya admitted at?\n\nAnswer: [/INST]"

    monkeypatch.setattr("secure_rag.rag_pipeline.retrieve", fake_retrieve)
    monkeypatch.setattr("secure_rag.rag_pipeline.generate_answer", fake_generate_answer)

    result = list(rag_answer("where was priya admitted at?", object(), ["unused"]))

    assert result == ["[ORG_MASKED]"]
