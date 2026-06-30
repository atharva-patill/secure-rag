import typer#framework to buil cli apps
from rich.console import Console#console ui
from rich.panel import Panel

from .rag_pipeline import build_rag, rag_answer#connecting cli to backend logic

console = Console()#print => normal python output 
                    #console.print => better tui


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
        console.print(f"Error: {exc}", style="bold red")#clean error handling
        raise typer.Exit(code=1)


def main():#chat to cli command
    typer.run(chat)


if __name__ == "__main__":
    main()
