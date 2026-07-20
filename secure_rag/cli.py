import os
import re
import sys
import time
from pathlib import Path
import typer
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.markdown import Markdown

from .rag_pipeline import build_rag, rag_answer

console = Console()


def _is_upstream_error(exc: Exception) -> bool:
    """Return True if the exception originated from an external inference provider."""
    mod = type(exc).__module__
    return mod.startswith("openai") or mod.startswith("urllib")


def stream_string(text: str):
    """Yield tokens from a string to simulate a natural stream, preserving whitespace."""
    tokens = re.split(r"(\s+)", text)
    for token in tokens:
        if token:
            yield token


def clear_prompt_lines():
    """Move cursor up and clear the console lines occupied by the prompt."""
    # Move up 1 line to prompt, clear line; move up 1 line to separator, clear line
    sys.stdout.write("\033[A\033[2K\033[A\033[2K")
    sys.stdout.flush()


def chat(file_path: str):
    try:
        # Disable tqdm / progress bars during model loading
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BAR"] = "1"

        # 1. Header
        console.print("\n Secure RAG")
        console.print("[gray]────────────────────────────────────────────────────────[/gray]\n")

        # 2. Startup sequence
        console.print("Initializing Secure RAG...")
        vector_store, chunks = build_rag(file_path)

        # Elegant checkbox reveal
        time.sleep(0.15)
        console.print(" [green]✓[/green] Embedding model")
        time.sleep(0.15)
        console.print(" [green]✓[/green] Vector database")
        time.sleep(0.15)
        console.print(" [green]✓[/green] Retriever")
        time.sleep(0.15)
        console.print(" [green]✓[/green] LLM")
        time.sleep(0.1)
        console.print("[bold green]Ready.[/bold green]\n")

        # Minimal Footer info displayed once
        console.print("[gray]Ctrl+C Quit • Ctrl+L Clear[/gray]\n")

        # First message from Assistant
        console.print("[bold blue]Assistant[/bold blue]\n")


        while True:
            # Keep prompt fixed at the bottom with horizontal separator
            try:
                query = console.input("[gray]────────────────────────────────────────[/gray]\n[bold]❯ [/bold]")
            except (KeyboardInterrupt, EOFError):
                console.print("\n[bold red]Exiting[/bold red]")
                break

            # Handle commands
            if not query.strip():
                continue

            if query.strip().lower() in ["exit", "quit"]:
                console.print("\n[bold red]Exiting[/bold red]")
                break

            if query.strip().lower() in ["clear", "cls"]:
                console.clear()
                console.print("\n Secure RAG")
                console.print("[gray]────────────────────────────────────────────────────────[/gray]\n")
                console.print("[bold blue]Assistant[/bold blue]\n")
                console.print("Ready whenever you are.\n")
                continue

            # Clear the input prompt from screen to render You message clean
            clear_prompt_lines()

            # Render You message
            console.print("[bold]You[/bold]\n")
            console.print(f"{query}\n")

            # Render Assistant message label
            console.print("[bold blue]Assistant[/bold blue]\n")

            # Show a very small animated status with Spinner (dots preferred)
            # Only the spinner should animate.
            # As soon as first streamed token arrives: remove spinner, stream response.
            response_text = ""
            with Live(Spinner("dots", text="Thinking...", style="blue"), console=console, transient=True) as live:
                response_gen = rag_answer(query, vector_store, chunks)
                response_text = next(response_gen)

            # Stream response beautifully using Live Markdown
            accumulated = ""
            with Live(Markdown(""), console=console, auto_refresh=False) as live_md:
                for token in stream_string(response_text):
                    accumulated += token
                    live_md.update(Markdown(accumulated), refresh=True)
                    time.sleep(0.01)

            # 4. Sources section
            console.print("\n[gray]Sources[/gray]")
            source_name = Path(file_path).name
            console.print(f"[gray]• {source_name}[/gray]\n")

    except Exception as exc:
        if _is_upstream_error(exc):
            msg = (
                "\n[bold red]Generation failed.[/bold red]\n\n"
                "[bold]Reason:[/bold]\n"
                "The external inference provider reported an error.\n\n"
                "[bold]Details:[/bold]\n"
                f"{exc}\n\n"
                "[bold]Suggestions:[/bold]\n"
                "  \u2022 Retry in a few moments.\n"
                "  \u2022 Choose a different model.\n"
                "  \u2022 Use the local Ollama backend for offline inference."
            )
            console.print(msg)
        else:
            console.print(f"\nError: {exc}", style="bold red")
        raise typer.Exit(code=1)


def main():
    typer.run(chat)


if __name__ == "__main__":
    main()
