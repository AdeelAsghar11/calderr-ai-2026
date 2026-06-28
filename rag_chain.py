import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

load_dotenv()

console = Console()

# ── Step 1: Load document ────────────────────────────────────────
console.print(Rule("[bold cyan]RAG Pipeline Setup[/bold cyan]"))
console.print("[dim]Step 1: Loading document...[/dim]")

loader = TextLoader("./documents/ai_notes.txt", encoding="utf-8")
docs = loader.load()
console.print(f"[green]✓[/green] Loaded {len(docs)} document(s)")

# ── Step 2: Split into chunks ────────────────────────────────────
console.print("[dim]Step 2: Splitting into chunks...[/dim]")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # max characters per chunk
    chunk_overlap=50,     # overlap so context isn't lost at boundaries
)
chunks = splitter.split_documents(docs)
console.print(f"[green]✓[/green] Split into {len(chunks)} chunks")

# ── Step 3: Create embeddings ────────────────────────────────────
console.print("[dim]Step 3: Loading embedding model (first run downloads ~90MB)...[/dim]")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",   # small, fast, good quality
    model_kwargs={"device": "cpu"},
)
console.print("[green]✓[/green] Embedding model ready")

# ── Step 4: Store in ChromaDB (in-memory) ───────────────────────
console.print("[dim]Step 4: Building vector store...[/dim]")

vectorstore = Chroma.from_documents(chunks, embeddings)
console.print(f"[green]✓[/green] Vector store built with {len(chunks)} vectors")

# ── Step 5: Create retriever ─────────────────────────────────────
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # return top 3 chunks

# ── Step 6: Build RAG chain ──────────────────────────────────────
console.print("[dim]Step 6: Building RAG chain...[/dim]")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer the question using ONLY the context below. "
     "If the answer is not in the context, say 'I don't have that information.' "
     "Be concise and accurate.\n\nContext:\n{context}"),
    ("user", "{question}")
])

def format_docs(docs) -> str:
    """Join retrieved chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

console.print("[green]✓[/green] RAG chain ready\n")
console.print(Rule("[bold cyan]Ask Questions[/bold cyan]"))
console.print("[dim]Type your question or 'exit' to quit[/dim]\n")

# ── Step 7: Interactive Q&A loop ─────────────────────────────────
while True:
    try:
        question = console.input("[bold green]You:[/bold green] ").strip()
    except (KeyboardInterrupt, EOFError):
        break

    if not question:
        continue
    if question.lower() == "exit":
        break

    with console.status("[dim]Retrieving + generating...[/dim]", spinner="dots"):
        # Show what was retrieved
        retrieved = retriever.invoke(question)
        answer = chain.invoke(question)

    console.print("\n[bold blue]Answer:[/bold blue]")
    console.print(Panel(answer, border_style="blue"))

    console.print("[dim]Retrieved chunks:[/dim]")
    for i, doc in enumerate(retrieved, 1):
        preview = doc.page_content[:120].replace("\n", " ")
        console.print(f"  [dim]{i}. {preview}...[/dim]")
    console.print()

console.print("[dim]Done.[/dim]")
