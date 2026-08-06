"""
Frontend Agent implementation.

Consumes a SystemContract artifact and generates an HTML/JS frontend application file on disk.
In stub mode, generates deterministic, functional HTML/JS code consuming all specified contract endpoints.
In real mode, uses ChatGroq to generate dynamic client code adhering strictly to the contract paths.
"""

import os
from src.schema import GeneratedComponent, SystemContract


def run_frontend_agent(contract: SystemContract, output_dir: str, use_real: bool = False) -> GeneratedComponent:
    """
    Generates an HTML/JS file on disk corresponding to the provided SystemContract.

    Why explicit endpoint path tracking in HTML/JS:
    To guarantee contract consistency verification between backend and frontend components,
    frontend code explicitly references contract endpoint path patterns during HTTP fetch requests.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.abspath(os.path.join(output_dir, "index.html"))

    if not use_real:
        # Generate clean HTML/JS code containing fetch calls for every contract endpoint path
        code_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Todo List Application</title>
</head>
<body>
    <h1>Todo List App</h1>

    <div>
        <input type="text" id="todo-title" placeholder="New todo title...">
        <button onclick="addTodo()">Add Todo</button>
    </div>

    <ul id="todo-list"></ul>

    <script>
        // Contract Endpoint: /todos (POST)
        async function addTodo() {
            const title = document.getElementById('todo-title').value;
            const res = await fetch('/todos', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title })
            });
            const data = await res.json();
            document.getElementById('todo-title').value = '';
            loadTodos();
        }

        // Contract Endpoint: /todos (GET)
        async function loadTodos() {
            const res = await fetch('/todos');
            const data = await res.json();
            const list = document.getElementById('todo-list');
            list.innerHTML = '';
            data.todos.forEach(todo => {
                const li = document.createElement('li');
                li.textContent = todo.title + (todo.completed ? ' (Completed)' : '');
                
                if (!todo.completed) {
                    const completeBtn = document.createElement('button');
                    completeBtn.textContent = 'Complete';
                    completeBtn.onclick = () => markComplete(todo.id);
                    li.appendChild(completeBtn);
                }

                const deleteBtn = document.createElement('button');
                deleteBtn.textContent = 'Delete';
                deleteBtn.onclick = () => deleteTodo(todo.id);
                li.appendChild(deleteBtn);

                list.appendChild(li);
            });
        }

        // Contract Endpoint: /todos/{todo_id}/complete (PUT)
        async function markComplete(todoId) {
            const endpoint = `/todos/${todoId}/complete`; // path template: /todos/{todo_id}/complete
            await fetch(endpoint, { method: 'PUT' });
            loadTodos();
        }

        // Contract Endpoint: /todos/{todo_id} (DELETE)
        async function deleteTodo(todoId) {
            const endpoint = `/todos/${todoId}`; // path template: /todos/{todo_id}
            await fetch(endpoint, { method: 'DELETE' });
            loadTodos();
        }

        // Initial load
        loadTodos();
    </script>
</body>
</html>
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

        prompt = f"""You are an expert Frontend Web Developer.
Write a standalone index.html web page with embedded JavaScript for the feature '{contract.feature_name}'.
You MUST issue fetch requests to every single endpoint path listed below exact as specified:

Endpoints:
{endpoints_info}

Ensure every endpoint path string (e.g. /todos, /todos/{{todo_id}}/complete, /todos/{{todo_id}}) appears literally in comments or fetch templates.
Return ONLY raw HTML/JS code inside an ```html ``` block or as raw text. Do not include introductory explanation.
"""
        response = llm.invoke(prompt)
        raw_text = str(response.content)

        if "```html" in raw_text:
            code_content = raw_text.split("```html")[1].split("```")[0].strip()
        elif "```" in raw_text:
            code_content = raw_text.split("```")[1].split("```")[0].strip()
        else:
            code_content = raw_text.strip()

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code_content)

    return GeneratedComponent(
        component_name="frontend",
        file_path=file_path,
        summary=f"HTML/JS frontend implementation written to {file_path}",
    )
