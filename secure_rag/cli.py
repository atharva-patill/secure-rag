import typer#framework to buil cli apps
from rich.console import Console#console ui
from rich.panel import Panel

from .rag_pipeline import build_rag, rag_answer#connecting cli to backend logic

console = Console()#print => normal python output 
                    #console.print => better tui


def chat(file_path: str):#main cli command
    try:#prevent crashes
        console.print(Panel("Secure RAG Chat\nType 'exit' to quit", style="bold green"))#tui header
        vector_store, chunks = build_rag(file_path)

        while True:
            query = console.input("[bold cyan]You:[/bold cyan] ")
            if query.lower() in ["exit", "quit"]:
                console.print("\nExiting", style="bold red")
                break

            console.print("[yellow]Thinking...[/yellow]")
            response = rag_answer(query, vector_store, chunks)

            console.print("[bold green]LLM:[/bold green] ", end="")
            for token in response:
                console.print(token, end="")
            console.print("\n")
    except Exception as exc:
        console.print(f"Error: {exc}", style="bold red")
        raise typer.Exit(code=1)


def main():
    typer.run(chat)


if __name__ == "__main__":
    main()
