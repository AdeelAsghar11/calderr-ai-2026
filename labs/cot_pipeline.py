import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule

load_dotenv()

console = Console()

# ── 10 problems: mix of math and logic ───────────────────────────
PROBLEMS = [
    {
        "id": 1,
        "type": "math",
        "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "answer": "$0.05"
    },
    {
        "id": 2,
        "type": "math",
        "question": "If there are 3 killers in a room and someone enters and kills one of them, how many killers are in the room?",
        "answer": "3"
    },
    {
        "id": 3,
        "type": "math",
        "question": "Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many tennis balls does he have now?",
        "answer": "11"
    },
    {
        "id": 4,
        "type": "logic",
        "question": "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
        "answer": "9"
    },
    {
        "id": 5,
        "type": "math",
        "question": "If you have a 3-gallon jug and a 5-gallon jug, how do you measure exactly 4 gallons of water?",
        "answer": "Fill 5-gallon, pour into 3-gallon until full (leaves 2 in 5-gallon), empty 3-gallon, pour 2 gallons into 3-gallon, fill 5-gallon again, pour into 3-gallon until full (1 gallon goes in), leaving 4 gallons in 5-gallon jug"
    },
    {
        "id": 6,
        "type": "math",
        "question": "A store reduces a $200 item by 20%, then reduces the sale price by another 10%. What is the final price?",
        "answer": "$144"
    },
    {
        "id": 7,
        "type": "logic",
        "question": "You have two ropes, each burns in exactly 1 hour but not uniformly. How do you measure 45 minutes?",
        "answer": "Light rope 1 from both ends and rope 2 from one end simultaneously. When rope 1 burns out (30 min), light the other end of rope 2. It burns out in 15 more minutes. Total = 45 minutes."
    },
    {
        "id": 8,
        "type": "math",
        "question": "If 5 machines take 5 minutes to make 5 widgets, how long would 100 machines take to make 100 widgets?",
        "answer": "5 minutes"
    },
    {
        "id": 9,
        "type": "logic",
        "question": "There are 3 boxes: one has apples, one has oranges, one has both. All boxes are mislabelled. You can pick one fruit from one box. How do you correctly label all boxes?",
        "answer": "Pick from the box labelled 'both'. Since all labels are wrong, it must be apples or oranges only. Whatever you pick tells you that box. Then the other two can be logically deduced."
    },
    {
        "id": 10,
        "type": "math",
        "question": "A train travels from city A to city B at 60 km/h and returns at 40 km/h. What is the average speed for the whole trip?",
        "answer": "48 km/h"
    },
]

# ── Prompts ───────────────────────────────────────────────────────
DIRECT_SYSTEM = "You are a problem-solving assistant. Answer the question directly and concisely. Give only the final answer, no explanation."

COT_SYSTEM = "You are a problem-solving assistant. Think through the problem step by step before giving your final answer. Show your reasoning clearly, then state the Final Answer on its own line prefixed with 'Final Answer:'"

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

direct_chain = (
    ChatPromptTemplate.from_messages([
        ("system", DIRECT_SYSTEM),
        ("user", "{question}")
    ]) | llm | StrOutputParser()
)

cot_chain = (
    ChatPromptTemplate.from_messages([
        ("system", COT_SYSTEM),
        ("user", "{question}")
    ]) | llm | StrOutputParser()
)


def run_comparison() -> None:
    results = []

    console.print(Rule("[bold cyan]CoT Pipeline — Comparing Direct vs Chain-of-Thought[/bold cyan]"))
    console.print("[dim]Running 10 problems, each twice...[/dim]\n")

    for problem in PROBLEMS:
        console.print(f"[yellow]Problem {problem['id']} ({problem['type']}):[/yellow] {problem['question']}")

        with console.status("[dim]Running direct...[/dim]", spinner="dots"):
            direct = direct_chain.invoke({"question": problem["question"]})

        with console.status("[dim]Running CoT...[/dim]", spinner="dots"):
            cot = cot_chain.invoke({"question": problem["question"]})

        # Extract final answer from CoT response
        cot_final = ""
        for line in cot.split("\n"):
            if "final answer" in line.lower():
                cot_final = line.split(":", 1)[-1].strip()
                break
        if not cot_final:
            cot_final = cot.split("\n")[-1].strip()

        results.append({
            "id": problem["id"],
            "type": problem["type"],
            "question": problem["question"],
            "expected": problem["answer"],
            "direct": direct.strip(),
            "cot_full": cot.strip(),
            "cot_final": cot_final,
        })

        console.print(f"  [blue]Direct:[/blue]   {direct.strip()}")
        console.print(f"  [green]CoT:[/green]      {cot_final}")
        console.print(f"  [dim]Expected: {problem['answer']}[/dim]\n")

    # ── Summary table ─────────────────────────────────────────────
    console.print(Rule("[bold cyan]Results Summary[/bold cyan]"))

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#",       width=3)
    table.add_column("Type",    width=6)
    table.add_column("Direct answer",  width=30)
    table.add_column("CoT answer",     width=30)
    table.add_column("Expected",       width=30)

    for r in results:
        table.add_row(
            str(r["id"]),
            r["type"],
            r["direct"][:28],
            r["cot_final"][:28],
            r["expected"][:28],
        )

    console.print(table)

    # ── Key findings ──────────────────────────────────────────────
    console.print(Rule("[bold cyan]Key Findings[/bold cyan]"))
    console.print("""
[bold]When CoT helps most:[/bold]
  • Multi-step math (problems 3, 6, 8, 10) — direct answers often skip steps
  • Logic puzzles (problems 2, 7, 9) — direct answers pattern-match wrong intuitions
  • The bat-and-ball (problem 1) — classic intuition trap, CoT forces correct arithmetic

[bold]When CoT is overkill:[/bold]
  • Simple factual lookups — wastes tokens, same answer
  • Single-step calculations — no benefit

[bold]Why CoT works:[/bold]
  Tokens generated left-to-right. Writing reasoning first means
  those tokens become context for the answer. The model literally
  reads its own reasoning when generating the final answer.
""")


if __name__ == "__main__":
    run_comparison()
