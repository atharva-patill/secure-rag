from rich.console import Console    #for styled outputs
from rich.panel import Panel        #panel draws boxes
from rich.prompt import Prompt      #for inputs

from rag_pipeline import rag_answer

console=Console()   #replacing print (print=>basic python output)
                    #console.print => styled ui
def run_chat(): #main fn
    console.print(#renders the box
        Panel(#creates box
            "Secure RAG Terminal Chat\nType 'exit' to quit",
            style="bold green"
        )
    )
    while True:#chat is in infinite loop
        #user input 
        user_input=Prompt.ask("\n[bold cyan]You[/bold cyan]")
        if user_input.lower()in["exit","quit"]:#exit/quit to exit the loop
            console.print("\n Exiting...\n",style="bold red")
            break

        #show buffering/thinking
        console.print("[yellow] Thinking..[/yellow]")

        #response
        response=rag_answer(user_input)#calls rag_pipeline
        if not isinstance(response,str):#converting genrator to string
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