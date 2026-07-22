import pytest
import typer
from typer.testing import CliRunner
from secure_rag.cli import chat, _is_upstream_error


# Create a temporary Typer app for testing
app = typer.Typer()
app.command()(chat)


def test_is_upstream_error():
    class MockOpenAIError(Exception):
        pass
    MockOpenAIError.__module__ = "openai.error"
    assert _is_upstream_error(MockOpenAIError()) is True

    class NormalError(Exception):
        pass
    NormalError.__module__ = "builtins"
    assert _is_upstream_error(NormalError()) is False


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Show this message and exit" in result.output


def test_cli_chat_exit(tmp_path, monkeypatch):
    # Mock build_rag and rag_answer to avoid heavy loading and API dependencies
    monkeypatch.setattr("secure_rag.cli.build_rag", lambda path: (object(), ["chunk1"]))

    runner = CliRunner()
    # Provide 'exit' to exit immediately
    result = runner.invoke(app, [str(tmp_path / "dummy.txt")], input="exit\n")
    assert result.exit_code == 0
    assert "Secure RAG" in result.output
    assert "Initializing" in result.output
    assert "Ready" in result.output
    assert "Exiting" in result.output


def test_cli_chat_query(tmp_path, monkeypatch):
    # Mock build_rag and rag_answer
    monkeypatch.setattr("secure_rag.cli.build_rag", lambda path: (object(), ["chunk1"]))

    def mock_rag_answer(query, vs, chunks):
        yield "This is a mock answer."

    monkeypatch.setattr("secure_rag.cli.rag_answer", mock_rag_answer)

    runner = CliRunner()
    # Query: 'what' followed by 'exit'
    result = runner.invoke(app, [str(tmp_path / "dummy.txt")], input="what\nexit\n")
    assert result.exit_code == 0
    assert "Secure RAG" in result.output
    assert "This is a mock answer." in result.output
    assert "Sources" in result.output
