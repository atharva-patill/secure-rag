from types import SimpleNamespace

from secure_rag import generator


def _stream_chunk(content):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content),
            )
        ]
    )


def test_generate_answer_flushes_short_streamed_answer(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            return iter([_stream_chunk("chest"), _stream_chunk(" pain")])

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions()),
    )

    monkeypatch.setattr(generator, "LLM_PROVIDER", "hf")
    monkeypatch.setattr(generator, "get_client", lambda: fake_client)

    response = list(generator.generate_answer("context", "question"))

    assert response == ["chest pain"]
