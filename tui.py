from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from rag_pipeline import rag_answer

console=Console()

def run_chat():
    console.print(
        Panel(
            "Secure RAG Terminal Chat\nType 'exit' to quit",
            style="bold green"
        )
    )
    while True:
        #user input 
        user_input=Prompt.ask("\n[bold cyan]You[/bold cyan]")
        if user_input.lower()in["exit","quit"]:
            console.print("\n Exiting...\n",style="bold red")
            break

        #show buffering/thinking
        console.print("[yellow] thinking..[/yellow]")

        #response
        response=rag_answer(user_input)
        if not isinstance(response,str):
            response="".join(response)
        #print response
        console.print(
            Panel(
                response,
                title="LLM",
                style="bold magenta"
            )
        )
if __name__=="__main__":
    run_chat()