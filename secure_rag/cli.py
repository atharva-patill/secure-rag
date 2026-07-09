import typer#framework to buil cli apps
from rich.console import Console#console ui
from rich.panel import Panel
from rich import box

from .rag_pipeline import build_rag, rag_answer#connecting cli to backend logic

console = Console()#print => normal python output 
                    #console.print => better tui


def _is_upstream_error(exc: Exception) -> bool:
    """Return True if the exception originated from an external inference provider."""
    mod = type(exc).__module__
    return mod.startswith("openai") or mod.startswith("urllib")


def chat(file_path: str):#main cli command
    try:#prevent crashes
        console.print(Panel("Secure RAG Chat\nType 'exit' to quit", style="bold green"))#tui header
        vector_store, chunks = build_rag(file_path)#data flow : load_data() → split_into_records() → mask_text() → chunk_record() → embed_chunks() → VectorStore

        while True:#infinite loop until exited
            query = console.input("[bold cyan]You:[/bold cyan] ")#query == user input
            if query.lower() in ["exit", "quit"]:#exit condition 
                console.print("\nExiting", style="bold red")#exit message
                break#break the loop

            console.print("[yellow]Thinking...[/yellow]")#think text with rich ui
            response = rag_answer(query, vector_store, chunks)#calling rag_pipeline

            console.print("[bold green]LLM:[/bold green] ")#streaming response , returning token-to-token
            for token in response:
                console.print(token, end="", markup=False)
            console.print("\n")
    except Exception as exc:#exception handling
        if _is_upstream_error(exc):
            msg = (
                "[bold red]Generation failed.[/bold red]\n\n"
                "[bold]Reason:[/bold]\n"
                "The external inference provider reported an error.\n\n"
                "[bold]Details:[/bold]\n"
                f"{exc}\n\n"
                "[bold]Suggestions:[/bold]\n"
                "  \u2022 Retry in a few moments.\n"
                "  \u2022 Choose a different model.\n"
                "  \u2022 Use the local Ollama backend for offline inference."
            )
            console.print(Panel(msg, title="Upstream Error", border_style="red", box=box.SQUARE))
        else:
            console.print(f"Error: {exc}", style="bold red")#clean error handling
        raise typer.Exit(code=1)


def main():#chat to cli command
    typer.run(chat)


if __name__ == "__main__":
    main()
