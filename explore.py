import os
import sys
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

question = "Explain what a neural network is in 2 sentences."

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("user", "{question}")
])

# ── 1. Model comparison ──────────────────────────────────────────
models = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "qwen/qwen3.6-27b",
]

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for model in models:
    llm = ChatGroq(model=model, temperature=0.7, api_key=os.getenv("GROQ_API_KEY"))
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question})
    print(f"\n[{model}]\n{response}")

# ── 2. Temperature experiment ────────────────────────────────────
print("\n" + "=" * 60)
print("TEMPERATURE EXPERIMENT (same model, same question)")
print("=" * 60)

temps = [0, 1, 2]
question2 = "Write a creative one-sentence description of the moon."

for temp in temps:
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=temp, api_key=os.getenv("GROQ_API_KEY"))
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question2})
    print(f"\n[temperature={temp}]\n{response}")