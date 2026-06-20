import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI engineering assistant."),
    ("user", "{question}")
])

chain = prompt | llm | StrOutputParser()

response = chain.invoke({"question": "What is an AI agent and why does it matter?"})
print(response)

print("\n--- Streaming ---")
for chunk in chain.stream({"question": "What is an AI agent?"}):
    print(chunk, end="", flush=True)