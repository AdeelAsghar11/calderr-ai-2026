"""
cli.py — Typer + Rich CLI interface for the Memory-Augmented Chatbot.

Provides interactive chat with cross-session episodic + semantic recall.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    # pyrefly: ignore [missing-import]
    from .chatbot import make_real_chatbot, make_stub_chatbot
except ImportError:
    # pyrefly: ignore [missing-import]
    from chatbot import make_real_chatbot, make_stub_chatbot


app = typer.Typer(help="Lab 6.1 Memory-Augmented Chatbot CLI")
console = Console()


@app.command()
def chat(
    real: bool = typer.Option(
        False, "--real", help="Use real Groq LLM API instead of offline stub"
    ),
    db_path: Optional[str] = typer.Option(
        None, "--db-path", help="Path to SQLite episodic database"
    ),
    chroma_path: Optional[str] = typer.Option(
        None, "--chroma-path", help="Path to ChromaDB storage directory"
    ),
    session_id: Optional[str] = typer.Option(
        None, "--session-id", help="Session ID (defaults to new random UUID)"
    ),
) -> None:
    """
    Start an interactive memory-augmented chat session.
    """
    if not session_id:
        session_id = f"session_{uuid.uuid4().hex[:8]}"

    if real:
        chatbot = make_real_chatbot(db_path=db_path, chroma_path=chroma_path)
        mode_str = "[bold green]Real (Groq LLM)[/bold green]"
    else:
        chatbot = make_stub_chatbot(db_path=db_path, chroma_path=chroma_path)
        mode_str = "[bold yellow]Offline Stub[/bold yellow]"

    console.print(
        Panel(
            f"Started Memory-Augmented Chatbot Session\n"
            f"Session ID: [cyan]{session_id}[/cyan]\n"
            f"Mode: {mode_str}",
            title="Lab 6.1 Memory Chatbot",
            border_style="blue",
        )
    )

    console.print("Type [bold red]'exit'[/bold red] or [bold red]'quit'[/bold red] to end the session.\n")

    while True:
        try:
            user_input = console.input("[bold cyan]User>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            console.print("[dim]Session ended.[/dim]")
            break

        # Process turn
        reply, retrieved = chatbot.process_user_turn(
            session_id=session_id,
            user_message=user_input,
        )

        # Print retrieved memories in a dim table if any exist
        if retrieved:
            table = Table(
                title=f"Retrieved Cross-Session Memories (Excluding '{session_id}')",
                style="dim",
                header_style="dim bold",
            )
            table.add_column("Session", style="dim cyan")
            table.add_column("Role", style="dim green")
            table.add_column("Content", style="dim white")
            table.add_column("Recency", style="dim yellow", justify="right")
            table.add_column("Relevance", style="dim yellow", justify="right")
            table.add_column("Composite", style="dim magenta", justify="right")

            for mem in retrieved:
                table.add_row(
                    mem.entry.session_id,
                    mem.entry.role,
                    mem.entry.content[:50] + ("..." if len(mem.entry.content) > 50 else ""),
                    f"{mem.recency_score:.4f}",
                    f"{mem.relevance_score:.4f}",
                    f"{mem.composite_score:.4f}",
                )

            console.print(table)
        else:
            console.print("[dim]No prior cross-session memories retrieved.[/dim]")

        # Print assistant reply
        console.print(
            Panel(
                reply,
                title="[bold green]Assistant[/bold green]",
                border_style="green",
            )
        )
        console.print()


if __name__ == "__main__":
    app()
