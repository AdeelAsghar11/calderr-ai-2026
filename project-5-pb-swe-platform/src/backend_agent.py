"""
Backend Agent implementation.

Consumes a SystemContract artifact and generates a fully functioning, syntactically valid FastAPI backend file on disk.
In stub mode, generates deterministic production-quality Python code. In real mode, queries ChatGroq to dynamically
produce code adhering strictly to every specified endpoint path and payload contract.
"""

import os
import ast
from src.schema import GeneratedComponent, SystemContract


def run_backend_agent(contract: SystemContract, output_dir: str, use_real: bool = False) -> GeneratedComponent:
    """
    Generates a FastAPI Python file on disk corresponding to the provided SystemContract.

    Why writing directly to disk is required:
    A summary or in-memory string describing code is not executable software.
    Writing source files directly to disk allows syntax parsers (ast.parse) and automated integration tests
    to inspect real filesystem artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.abspath(os.path.join(output_dir, "backend.py"))

    if not use_real:
        # Deterministic FastAPI backend Python code implementing contract endpoints
        code_content = '''"""
Generated FastAPI Backend Application.
Automated generation based on SystemContract specifications.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Todo List API")

class TodoCreate(BaseModel):
    title: str

class TodoItem(BaseModel):
    id: int
    title: str
    completed: bool

todos_db = []

@app.post("/todos", response_model=TodoItem, status_code=201)
def create_todo(todo: TodoCreate):
    new_todo = {
        "id": len(todos_db) + 1,
        "title": todo.title,
        "completed": False
    }
    todos_db.append(new_todo)
    return new_todo

@app.get("/todos")
def list_todos():
    return {"todos": todos_db}

@app.put("/todos/{todo_id}/complete")
def complete_todo(todo_id: int):
    for todo in todos_db:
        if todo["id"] == todo_id:
            todo["completed"] = True
            return todo
    raise HTTPException(status_code=404, detail="Todo not found")

@app.delete("/todos/{todo_id}")
def delete_todo(todo_id: int):
    global todos_db
    initial_len = len(todos_db)
    todos_db = [t for t in todos_db if t["id"] != todo_id]
    if len(todos_db) == initial_len:
        raise HTTPException(status_code=404, detail="Todo not found")
    return {"status": "deleted"}
'''
    else:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is required for real mode execution.")

        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=api_key,
        )

        endpoints_info = "\n".join(
            [f"- Method: {e.method}, Path: {e.path}, Request: {e.request_fields}, Response: {e.response_fields}" for e in contract.endpoints]
        )

        prompt = f"""You are an expert Python Backend Developer.
Write executable, valid Python FastAPI code for the feature '{contract.feature_name}'.
You MUST implement every single endpoint path listed below exact as specified:

Endpoints:
{endpoints_info}

Return ONLY raw Python code inside a ```python ``` block or as raw text. Do not include introductory text.
"""
        response = llm.invoke(prompt)
        raw_text = str(response.content)
        # Extract code from markdown blocks if present
        if "```python" in raw_text:
            code_content = raw_text.split("```python")[1].split("```")[0].strip()
        elif "```" in raw_text:
            code_content = raw_text.split("```")[1].split("```")[0].strip()
        else:
            code_content = raw_text.strip()

    # Validate Python syntax before saving
    try:
        ast.parse(code_content)
    except SyntaxError as e:
        raise RuntimeError(f"Generated backend code failed Python AST syntax parsing: {e}")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code_content)

    return GeneratedComponent(
        component_name="backend",
        file_path=file_path,
        summary=f"FastAPI backend implementation written to {file_path}",
    )
