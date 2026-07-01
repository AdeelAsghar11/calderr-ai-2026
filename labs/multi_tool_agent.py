import os
import math
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

load_dotenv()

console = Console()

# ── Mock news database ───────────────────────────────────────────
NEWS_DB: dict[str, str] = {
    "ai": "OpenAI released GPT-5 with a 1M token context window. Google DeepMind announced Gemini 2.0 Ultra. Anthropic raised $4B in Series E funding. Meta open-sourced Llama 4 with 405B parameters.",
    "pakistan": "Pakistan's IT exports reached $3.2B in 2025. Karachi Stock Exchange hit an all-time high. The government launched a National AI Policy framework. CPEC phase 2 infrastructure projects are underway.",
    "climate": "Global temperatures hit a record 1.6°C above pre-industrial levels. The EU announced a €500B green energy fund. Solar capacity worldwide doubled for the third consecutive year.",
    "crypto": "Bitcoin reached $120,000 in early 2026. Ethereum completed its second major protocol upgrade. The SEC approved 12 new Bitcoin ETFs. Binance settled its DOJ case for $4.3B.",
    "tech": "Apple unveiled Vision Pro 2 with standalone AI processing. Samsung launched foldable phones with satellite connectivity. NVIDIA's H200 GPU sold out globally through 2026.",
    "health": "A new mRNA vaccine for malaria entered phase 3 trials. WHO declared mpox no longer a global emergency. Ozempic generics became available in most countries.",
    "sports": "Pakistan won the T20 World Cup 2025. Manchester City won their 8th Premier League title. Novak Djokovic retired from professional tennis at age 38.",
    "economy": "US inflation stabilized at 2.1%. IMF raised global growth forecast to 3.4%. Pakistan's GDP grew 4.2% in FY2025. China's economy expanded 5.1%.",
}

def search_news(query: str) -> str:
    query_lower = query.lower()
    for key, value in NEWS_DB.items():
        if key in query_lower or any(word in query_lower for word in key.split()):
            return value
    # Try partial word match
    for key, value in NEWS_DB.items():
        if any(word in key for word in query_lower.split() if len(word) > 3):
            return value
    return f"No news found for '{query}'. Available topics: {', '.join(NEWS_DB.keys())}"


# ── Tool definitions ─────────────────────────────────────────────
@tool
def web_search_mock(query: str) -> str:
    """Search for the latest news and information on any topic.
    Use for: current events, news, facts about companies, people, or topics.
    Input: a short search query like 'AI news' or 'Pakistan economy'."""
    return search_news(query)


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.
    Use for: any arithmetic, percentages, or numerical calculations.
    Input: a valid Python math expression like '25 * 48' or '(100 * 1.15) / 2'.
    Supports: +, -, *, /, **, sqrt(), sin(), cos(), log(), pi, e."""
    try:
        result = eval(expression, {"__builtins__": {}}, vars(math))
        return f"{result}"
    except Exception as e:
        return f"Calculation error: {e}. Check the expression syntax."


@tool
def get_current_date(dummy: str = "") -> str:
    """Get today's current date, day, month, year and time.
    Use for: any question about today's date, current day, month, year, or time.
    No input needed."""
    now = datetime.now()
    return now.strftime("%A, %d %B %Y — %H:%M")


@tool
def classify_sentiment(text: str) -> str:
    """Classify the sentiment of a piece of text as positive, negative, or neutral.
    Use for: analyzing tone of news, reviews, statements, or any text.
    Input: the text to analyze. Returns: sentiment label + confidence + key words."""
    text_lower = text.lower()

    positive_words = ["growth", "record", "high", "success", "won", "approved",
                      "launched", "raised", "doubled", "stabilized", "expanded", "best"]
    negative_words = ["crisis", "fell", "decline", "loss", "failed", "collapsed",
                      "warning", "danger", "risk", "concern", "problem", "worst"]

    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)

    if pos_count > neg_count:
        sentiment = "POSITIVE"
        confidence = min(0.95, 0.6 + (pos_count * 0.05))
    elif neg_count > pos_count:
        sentiment = "NEGATIVE"
        confidence = min(0.95, 0.6 + (neg_count * 0.05))
    else:
        sentiment = "NEUTRAL"
        confidence = 0.70

    matched = [w for w in positive_words + negative_words if w in text_lower]
    return f"Sentiment: {sentiment} (confidence: {confidence:.0%}) | Key signals: {', '.join(matched[:5]) or 'none detected'}"


@tool
def summarize(text: str) -> str:
    """Summarize a long piece of text into 2-3 concise sentences.
    Use for: condensing news articles, long search results, or verbose content.
    Input: the text to summarize."""
    sentences = [s.strip() for s in text.replace(".", ". ").split(". ") if s.strip()]
    if len(sentences) <= 2:
        return text
    # Take first sentence + longest sentence (likely most informative) + last
    key_sentences = [sentences[0]]
    if len(sentences) > 2:
        longest = max(sentences[1:-1], key=len) if len(sentences) > 2 else sentences[1]
        key_sentences.append(longest)
    return ". ".join(key_sentences) + "."


# ── Agent loop ───────────────────────────────────────────────────
TOOLS = [web_search_mock, calculate, get_current_date, classify_sentiment, summarize]
TOOL_MAP = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """You are a helpful research assistant with access to 5 tools:

1. web_search_mock — search for news and current information
2. calculate — evaluate math expressions  
3. get_current_date — get today's date or year
4. classify_sentiment — analyze sentiment of text
5. summarize — condense long text into 2-3 sentences

Use tools whenever needed. You can call multiple tools in sequence.
Always give a clear final answer after using tools."""


def run_agent(question: str, llm) -> None:
    console.print(f"\n[bold white]Question:[/bold white] {question}")
    console.print(Rule(style="dim"))

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    step = 0
    max_steps = 8

    while step < max_steps:
        step += 1

        with console.status(f"[dim]Step {step}: calling LLM...[/dim]", spinner="dots"):
            response = llm.invoke(messages)

        messages.append(response)

        # ── No tool calls = final answer ──────────────────────────
        if not response.tool_calls:
            console.print(Panel(
                response.content,
                title="[green]Final Answer[/green]",
                border_style="green"
            ))
            return

        # ── Execute each tool call ────────────────────────────────
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]

            console.print(f"  [cyan]→ Tool:[/cyan] [bold]{tool_name}[/bold]")
            console.print(f"  [cyan]  Args:[/cyan] {tool_args}")

            if tool_name in TOOL_MAP:
                try:
                    result = TOOL_MAP[tool_name].invoke(tool_args)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {tool_name}"

            console.print(f"  [green]  Result:[/green] {str(result)[:200]}\n")

            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tc["id"]
            ))

    console.print("[red]Max steps reached.[/red]")


def main() -> None:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    ).bind_tools(TOOLS)

    console.print(Rule("[bold cyan]Lab 2.2 — Multi-Tool Research Agent[/bold cyan]"))

    # ── Tool registry display ─────────────────────────────────────
    table = Table(show_header=True, header_style="bold cyan", title="Available Tools")
    table.add_column("Tool", style="yellow")
    table.add_column("Purpose")
    for t in TOOLS:
        table.add_row(t.name, t.description.split("\n")[0])
    console.print(table)

    # ── Test queries — each targets different tools ───────────────
    test_queries = [
        "What is today's date?",
        "What's happening in AI news? Summarize it for me.",
        "What is 15% of 85000 plus 3200?",
        "Search for Pakistan economy news and tell me if the sentiment is positive or negative.",
        "What year is it and what is 2026 minus 1947?",
    ]

    console.print("\n[dim]Running 5 test queries...[/dim]")

    for query in test_queries:
        run_agent(query, llm)
        console.print()

    # ── Interactive mode ──────────────────────────────────────────
    console.print(Rule("[bold cyan]Interactive Mode[/bold cyan]"))
    console.print("[dim]Ask anything — type 'exit' to quit[/dim]\n")

    while True:
        try:
            question = console.input("[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not question or question.lower() == "exit":
            break
        run_agent(question, llm)


if __name__ == "__main__":
    main()