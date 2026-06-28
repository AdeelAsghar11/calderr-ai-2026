import os
import re
import math
from dotenv import load_dotenv
from groq import Groq
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

load_dotenv()

console = Console()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Mock facts database ──────────────────────────────────────────
FACTS_DB: dict[str, str] = {
    "pakistan population": "Pakistan has a population of approximately 240 million people.",
    "python creator": "Python was created by Guido van Rossum, first released in 1991.",
    "llm": "A Large Language Model (LLM) is a deep learning model trained on massive text datasets to understand and generate human-like text.",
    "langchain": "LangChain is a framework for building LLM-powered applications with chains, agents, and memory.",
    "groq": "Groq provides extremely fast LLM inference using custom LPU (Language Processing Unit) hardware.",
    "transformer": "The Transformer architecture was introduced in 2017 in 'Attention Is All You Need' and is the foundation of all modern LLMs.",
    "react pattern": "ReAct (Reasoning + Acting) is an agent pattern where the LLM alternates Thought, Action, and Observation steps.",
    "context window": "A context window is the maximum number of tokens an LLM can process at once. Llama 3 supports up to 128k tokens.",
    "temperature": "Temperature controls randomness in LLM output. 0 = deterministic, 1 = balanced, 2 = very random.",
    "comsats": "COMSATS University Islamabad (CUI) is a public research university in Pakistan with campuses across the country.",
    "calder r": "CalderR is a company running an Agentic AI Engineering Internship programme in 2026.",
}

# ── System prompt ────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a ReAct agent that solves problems step by step.

For EVERY response follow this EXACT format — no exceptions:

Thought: [your reasoning about what to do next]
Action: [one of: search, calculate, answer]
Action Input: [the input for the action]

Available tools:
- search      → look up a fact. Action Input = a short search query.
- calculate   → evaluate math. Action Input = a valid Python expression e.g. "240 / 10" or "25 * 48".
- answer      → give the final response. Use ONLY when you have enough information.
                Action Input = your complete answer to the user.

Rules:
- Always start with Thought:.
- Never skip Thought:.
- For multi-step problems, use search or calculate first, then answer.
- temperature=0 so be precise and consistent.
"""


# ── Tools ────────────────────────────────────────────────────────
def search(query: str) -> str:
    query_lower = query.lower().strip()
    for key, value in FACTS_DB.items():
        if key in query_lower or any(word in query_lower for word in key.split()):
            return value
    return "No information found in the database for that query."


def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, vars(math))
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


# ── Output parser ────────────────────────────────────────────────
def parse_output(text: str) -> tuple[str, str, str]:
    thought = ""
    action = ""
    action_input = ""

    thought_match = re.search(r"Thought:\s*(.+)", text)
    action_match = re.search(r"Action:\s*(.+)", text)
    input_match = re.search(r"Action Input:\s*(.+)", text)

    if thought_match:
        thought = thought_match.group(1).strip()
    if action_match:
        action = action_match.group(1).strip().lower()
    if input_match:
        action_input = input_match.group(1).strip()

    return thought, action, action_input


# ── Agent loop ───────────────────────────────────────────────────
def run_agent(question: str, max_steps: int = 6) -> None:
    console.print(Rule("[bold cyan]ReAct Agent[/bold cyan]"))
    console.print(f"\n[bold white]Question:[/bold white] {question}\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):
        console.print(f"[dim]── Step {step} ──[/dim]")

        # ── LLM call ─────────────────────────────────────────────
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0,
            max_tokens=300,
        )
        llm_output = response.choices[0].message.content.strip()

        # ── Parse ─────────────────────────────────────────────────
        thought, action, action_input = parse_output(llm_output)

        console.print(f"[yellow]Thought:[/yellow]      {thought}")
        console.print(f"[cyan]Action:[/cyan]       {action}")
        console.print(f"[cyan]Action Input:[/cyan] {action_input}")

        # ── Execute tool ──────────────────────────────────────────
        if action == "search":
            observation = search(action_input)
            console.print(f"[green]Observation:[/green]  {observation}\n")
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user", "content": f"Observation: {observation}\nContinue."})

        elif action == "calculate":
            observation = calculate(action_input)
            console.print(f"[green]Observation:[/green]  {observation}\n")
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user", "content": f"Observation: {observation}\nContinue."})

        elif action == "answer":
            console.print(Rule())
            console.print(Panel(
                f"[bold white]{action_input}[/bold white]",
                title="[green]Final Answer[/green]",
                border_style="green"
            ))
            return

        else:
            console.print(f"[red]Unknown action '{action}' — stopping.[/red]")
            return

    console.print("[red]Max steps reached without a final answer.[/red]")


# ── Test questions ───────────────────────────────────────────────
if __name__ == "__main__":
    questions = [
        "What is 25 multiplied by 48?",
        "Who created Python and when?",
        "What is the population of Pakistan divided by 10?",
        "What is LangChain and what is 100 divided by 4?",
    ]

    for q in questions:
        run_agent(q)
        console.print("\n")
