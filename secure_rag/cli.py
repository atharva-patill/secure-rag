import typer
from rich.console import Console
from rich.panel import Panel

from secure_rag import build_rag, rag_answer

app = typer.Typer()
console = Console()


@app.command()
def chat(file_path: str = typer.Argument(...)):
    """
    Start Secure RAG chat with a given data file
    """

    console.print(Panel("Secure RAG Chat\nType 'exit' to quit", style="bold green"))

    # Build RAG system
    vector_store, chunks = build_rag(file_path)

    while True:
        query = console.input("[bold cyan]You:[/bold cyan] ")

        if query.lower() in ["exit", "quit"]:
            console.print("\nExiting", style="bold red")
            break

        console.print("[yellow]Thinking...[/yellow]")

        response = rag_answer(query, vector_store, chunks)

        # STREAM OUTPUT
        console.print("[bold green]LLM:[/bold green] ", end="")

        for token in response:
            console.print(token, end="")

        console.print("\n")
if __name__ == "__main__":
    app()
def main():
    app()