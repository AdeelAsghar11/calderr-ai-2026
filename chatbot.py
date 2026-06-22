import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

load_dotenv()

console = Console()

# ── Domain system prompts ────────────────────────────────────────
DOMAINS: dict[str, str] = {
    "programming": (
        "You are an expert programming assistant. Help with code, debugging, "
        "architecture, and best practices. Always provide clean, typed Python "
        "examples when relevant. Refuse non-programming questions politely."
    ),
    "cooking": (
        "You are an expert chef and culinary assistant. Help with recipes, "
        "techniques, substitutions, and meal planning. Refuse non-cooking "
        "questions politely."
    ),
    "history": (
        "You are a knowledgeable history professor. Discuss historical events, "
        "figures, and their significance with accuracy and context. Refuse "
        "non-history questions politely."
    ),
    "general": (
        "You are a helpful, concise AI assistant. Answer clearly and accurately."
    ),
}

COMMANDS = "/help · /clear · /topic · /exit"


def show_banner(domain: str) -> None:
    console.print(Panel(
        f"[bold cyan]CalderR AI Assistant[/bold cyan]\n"
        f"[dim]Domain: [yellow]{domain}[/yellow] · {COMMANDS}[/dim]",
        border_style="cyan"
    ))


def show_help() -> None:
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Command", style="yellow")
    table.add_column("Description")
    table.add_row("/help",  "Show this help menu")
    table.add_row("/clear", "Clear conversation history")
    table.add_row("/topic", "Switch domain (programming / cooking / history / general)")
    table.add_row("/exit",  "Quit the assistant")
    console.print(table)


def show_topics() -> None:
    console.print("[bold]Available domains:[/bold] " +
                  " · ".join(f"[yellow]{d}[/yellow]" for d in DOMAINS))


def main() -> None:
    domain = "general"
    history: list[HumanMessage | AIMessage] = []

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    show_banner(domain)

    while True:
        try:
            user_input = console.input("\n[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/dim]")
            break

        if not user_input:
            continue

        # ── Commands ─────────────────────────────────────────────
        if user_input == "/exit":
            console.print("[dim]Bye![/dim]")
            break

        elif user_input == "/clear":
            history.clear()
            console.print("[yellow]History cleared.[/yellow]")
            continue

        elif user_input == "/help":
            show_help()
            continue

        elif user_input == "/topic":
            show_topics()
            new_domain = console.input("[bold]Enter domain:[/bold] ").strip().lower()
            if new_domain in DOMAINS:
                domain = new_domain
                history.clear()
                console.print(f"[yellow]Switched to [bold]{domain}[/bold]. History cleared.[/yellow]")
            else:
                console.print("[red]Unknown domain. Try: programming, cooking, history, general[/red]")
            continue

        # ── LLM call ─────────────────────────────────────────────
        history.append(HumanMessage(content=user_input))

        messages = [SystemMessage(content=DOMAINS[domain])] + history

        try:
            with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                response = llm.invoke(messages)

            ai_text: str = response.content
            history.append(AIMessage(content=ai_text))

            # ── Response output ───────────────────────────────────
            console.print("\n[bold blue]Assistant:[/bold blue]")
            console.print(Panel(Markdown(ai_text), border_style="blue"))

            # ── Token usage ───────────────────────────────────────
            usage = response.usage_metadata
            if usage:
                console.print(
                    f"[dim]Tokens — in: {usage['input_tokens']} · "
                    f"out: {usage['output_tokens']} · "
                    f"total: {usage['total_tokens']}[/dim]"
                )

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            history.pop()  # remove failed message from history


if __name__ == "__main__":
    main()