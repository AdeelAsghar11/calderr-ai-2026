import os
import json
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

# ── Test article ─────────────────────────────────────────────────
NEWS_ARTICLE = """
OpenAI has announced GPT-5, its most powerful language model to date. The new model 
demonstrates significant improvements in reasoning, coding, and multimodal capabilities 
compared to its predecessor GPT-4. In internal benchmarks, GPT-5 scores 87% on the 
MMLU test, compared to GPT-4's 86.4%. The model features a 1 million token context 
window and supports real-time voice and image inputs. OpenAI CEO Sam Altman stated 
that GPT-5 represents "a step change in AI capability." The model will be available 
to ChatGPT Plus subscribers immediately and via API next month. Pricing starts at 
$15 per million input tokens. Critics have raised concerns about potential misuse, 
while researchers note the model still struggles with complex mathematical proofs. 
The announcement comes amid increasing competition from Google's Gemini Ultra and 
Anthropic's Claude 3.5 Sonnet.
"""

# ── 5 System prompts ─────────────────────────────────────────────
PROMPTS: dict[str, str] = {

    "1_basic": (
        "Summarize the article."
    ),

    "2_role_based": (
        "You are a senior technology journalist at a major newspaper. "
        "Write a concise, professional summary of the article that captures "
        "the key facts, numbers, and implications for your readers."
    ),

    "3_few_shot": (
        "Summarize news articles in 2-3 sentences. Focus on: what happened, "
        "who is involved, and why it matters.\n\n"
        "Example input: Apple announced the iPhone 16 with a new A18 chip, "
        "improved cameras, and a starting price of $799. CEO Tim Cook called "
        "it 'the most advanced iPhone ever.'\n\n"
        "Example output: Apple unveiled the iPhone 16 featuring the A18 chip "
        "and upgraded cameras at $799. CEO Tim Cook emphasized the device's "
        "performance advancements, signaling Apple's continued focus on "
        "premium hardware.\n\n"
        "Now summarize the article provided."
    ),

    "4_chain_of_thought": (
        "Summarize the article. Before writing the summary, think through: "
        "1) What is the main announcement? "
        "2) What are the key numbers or facts? "
        "3) What are the implications? "
        "Then write a 2-3 sentence summary. "
        "Format: Thinking: [your reasoning]\nSummary: [final summary]"
    ),

    "5_structured": (
        "Summarize the article in exactly this format:\n\n"
        "HEADLINE: [one sentence, max 15 words]\n"
        "KEY FACTS: [3 bullet points, one fact each]\n"
        "BOTTOM LINE: [one sentence on why this matters]\n\n"
        "Be factual and concise. No opinions."
    ),
}

# ── Judge prompt ─────────────────────────────────────────────────
JUDGE_PROMPT = """You are an expert evaluator of news summaries.

Score the following summary on three criteria, each from 1 to 5:
- accuracy: Does it correctly represent the key facts from the article?
- conciseness: Is it brief and free of unnecessary words?
- tone: Is it professional, neutral, and appropriate?

Article:
{article}

Summary to evaluate:
{summary}

Respond ONLY with valid JSON in this exact format, nothing else:
{{"accuracy": <1-5>, "conciseness": <1-5>, "tone": <1-5>, "comment": "<one sentence>"}}
"""


def run_prompt(name: str, system_prompt: str, llm: ChatGroq) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Article:\n{article}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"article": NEWS_ARTICLE.strip()})


def judge_summary(article: str, summary: str, judge_llm: ChatGroq) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("user", JUDGE_PROMPT)
    ])
    chain = prompt | judge_llm | StrOutputParser()
    raw = chain.invoke({"article": article, "summary": summary})
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"accuracy": 0, "conciseness": 0, "tone": 0, "comment": "parse error"}


def score_bar(score: int, max_score: int = 5) -> str:
    filled = "█" * score
    empty = "░" * (max_score - score)
    return f"{filled}{empty} {score}/{max_score}"


def main() -> None:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    judge_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    console.print(Rule("[bold cyan]Lab 1.3 — Prompt A/B Test[/bold cyan]"))
    console.print("[dim]Task: Summarise a news article using 5 different prompt strategies[/dim]\n")

    results: list[dict] = []

    for name, system_prompt in PROMPTS.items():
        console.print(f"[yellow]Running prompt:[/yellow] {name}")

        with console.status("[dim]Generating summary...[/dim]", spinner="dots"):
            summary = run_prompt(name, system_prompt, llm)

        console.print(Panel(summary, title=f"[cyan]{name}[/cyan]", border_style="cyan"))

        with console.status("[dim]Judging...[/dim]", spinner="dots"):
            scores = judge_summary(NEWS_ARTICLE.strip(), summary, judge_llm)

        total = scores.get("accuracy", 0) + scores.get("conciseness", 0) + scores.get("tone", 0)
        results.append({
            "name": name,
            "summary": summary,
            "accuracy": scores.get("accuracy", 0),
            "conciseness": scores.get("conciseness", 0),
            "tone": scores.get("tone", 0),
            "total": total,
            "comment": scores.get("comment", ""),
        })
        console.print(
            f"  [green]Scores[/green] — "
            f"Accuracy: {scores.get('accuracy')}/5 · "
            f"Conciseness: {scores.get('conciseness')}/5 · "
            f"Tone: {scores.get('tone')}/5 · "
            f"Total: {total}/15\n"
        )

    # ── Results table ─────────────────────────────────────────────
    console.print(Rule("[bold cyan]Final Results[/bold cyan]"))

    results.sort(key=lambda x: x["total"], reverse=True)

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Rank", style="bold", width=4)
    table.add_column("Prompt", width=22)
    table.add_column("Accuracy", width=14)
    table.add_column("Conciseness", width=14)
    table.add_column("Tone", width=14)
    table.add_column("Total", width=7)
    table.add_column("Comment", width=40)

    for rank, r in enumerate(results, 1):
        table.add_row(
            str(rank),
            r["name"],
            score_bar(r["accuracy"]),
            score_bar(r["conciseness"]),
            score_bar(r["tone"]),
            f"[bold]{r['total']}/15[/bold]",
            r["comment"],
        )

    console.print(table)

    winner = results[0]
    console.print(f"\n[bold green]Best prompt:[/bold green] {winner['name']} "
                  f"with a total score of {winner['total']}/15")


if __name__ == "__main__":
    main()
