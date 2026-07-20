import os
import re
import sys
import time
from pathlib import Path

try:
    import select
    import termios
    import tty
except ImportError:  # pragma: no cover - Windows fallback
    select = None
    termios = None
    tty = None

import typer
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.markdown import Markdown
from rich.text import Text

from .rag_pipeline import build_rag, rag_answer

console = Console()

COMPOSER_GLYPH = "❯"
COMPOSER_PLACEHOLDER = "Ask a question about your documents..."


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


def _composer_width() -> int:
    return max(32, min(console.size.width, 88))


def _composer_lines(buffer: str):
    width = _composer_width()
    inner_width = width - 2
    prefix = f"  {COMPOSER_GLYPH} "
    text_width = max(1, inner_width - len(prefix) - 1)
    displayed = buffer[-text_width:] if buffer else COMPOSER_PLACEHOLDER[:text_width]

    top = Text("╭" + "─" * inner_width + "╮", style="grey39")
    middle = Text("│", style="grey39")
    middle.append("  ")
    middle.append(COMPOSER_GLYPH, style="bold cyan")
    middle.append(" ")
    middle.append(displayed, style="white" if buffer else "grey50")
    middle.append(" " * max(0, inner_width - len(prefix) - len(displayed)))
    middle.append("│", style="grey39")
    bottom = Text("╰" + "─" * inner_width + "╯", style="grey39")

    cursor_column = len("│") + len(prefix) + (len(displayed) if buffer else 0) + 1
    return top, middle, bottom, cursor_column


def _render_composer(buffer: str):
    top, middle, bottom, cursor_column = _composer_lines(buffer)
    console.print(top)
    console.print(middle)
    console.print(bottom)
    sys.stdout.write(f"\033[2A\033[{cursor_column}G")
    sys.stdout.flush()


def _clear_rendered_composer():
    """Clear the three-line composer while the cursor is on its input row."""
    sys.stdout.write("\r\033[1A\033[2K")
    sys.stdout.write("\033[1B\r\033[2K")
    sys.stdout.write("\033[1B\r\033[2K")
    sys.stdout.write("\033[2A\r")
    sys.stdout.flush()


def _drain_escape_sequence():
    if select is None:
        return
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 0.001)
        if not ready:
            return
        sys.stdin.read(1)


def _read_interactive_composer() -> str:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    buffer = []

    try:
        tty.setcbreak(fd)
        _render_composer("")

        while True:
            char = sys.stdin.read(1)

            if char in ("\r", "\n"):
                query = "".join(buffer)
                _clear_rendered_composer()
                return query

            if char == "\x03":
                _clear_rendered_composer()
                raise KeyboardInterrupt

            if char == "\x04" and not buffer:
                _clear_rendered_composer()
                raise EOFError

            if char in ("\x7f", "\b"):
                if buffer:
                    buffer.pop()
                    _clear_rendered_composer()
                    _render_composer("".join(buffer))
                continue

            if char == "\x1b":
                _drain_escape_sequence()
                continue

            if char.isprintable():
                buffer.append(char)
                _clear_rendered_composer()
                _render_composer("".join(buffer))
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def read_composer_input() -> str:
    if sys.stdin.isatty() and sys.stdout.isatty() and termios is not None and tty is not None:
        return _read_interactive_composer()

    return console.input("[gray]────────────────────────────────────────[/gray]\n[bold cyan]❯ [/bold cyan]")


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
        console.print("[gray]EXIT to Quit • CMD + K to Clear[/gray]\n")

        # First message from Assistant
        console.print("[bold blue]Assistant[/bold blue]\n")


        while True:
            try:
                query = read_composer_input()
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
