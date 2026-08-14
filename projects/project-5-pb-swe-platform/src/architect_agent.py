"""
Architect Agent implementation.

Takes a TaskList artifact from the PM Agent and designs a typed SystemContract.
The SystemContract specifies exact API paths, HTTP methods, payload fields, and shared data models.
This contract serves as the strict operational boundary for Backend and Frontend generation agents.
"""

import os
# pyrefly: ignore [missing-import]
from src.schema import EndpointSpec, SystemContract, TaskList


def run_architect_agent(task_list: TaskList, use_real: bool = False) -> SystemContract:
    """
    Executes the Architect Agent to design a SystemContract from a TaskList.

    Why SystemContract design is centralized:
    Decoupling architecture specification from code generation prevents backend/frontend API skew.
    Both generation agents consume this identical contract object rather than inferring endpoints independently.
    """
    if not use_real:
        # Stub implementation providing fixed SystemContract matching the todo feature spec
        return SystemContract(
            feature_name="Todo List API",
            endpoints=[
                EndpointSpec(
                    path="/todos",
                    method="POST",
                    request_fields={"title": "string"},
                    response_fields={"id": "integer", "title": "string", "completed": "boolean"},
                    description="Create a new todo item",
                ),
                EndpointSpec(
                    path="/todos",
                    method="GET",
                    request_fields={},
                    response_fields={"todos": "list[Todo]"},
                    description="Retrieve all todo items",
                ),
                EndpointSpec(
                    path="/todos/{todo_id}/complete",
                    method="PUT",
                    request_fields={},
                    response_fields={"id": "integer", "completed": "boolean"},
                    description="Mark a specific todo item as completed",
                ),
                EndpointSpec(
                    path="/todos/{todo_id}",
                    method="DELETE",
                    request_fields={},
                    response_fields={"status": "string"},
                    description="Delete a specific todo item",
                ),
            ],
            data_models={
                "Todo": {
                    "id": "integer",
                    "title": "string",
                    "completed": "boolean",
                }
            },
        )

    # Real mode execution via Groq structured output
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is required for real mode execution.")

    # pyrefly: ignore [missing-import]
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=api_key,
    )
    structured_llm = llm.with_structured_output(SystemContract)

    tasks_summary = "\n".join(
        [f"- {t.description} (Criteria: {', '.join(t.acceptance_criteria)})" for t in task_list.tasks]
    )

    prompt = f"""You are a Lead Software Architect.
Design a complete, typed SystemContract for the following tasks.
Specify exact REST endpoint paths (e.g. /todos, /todos/{{todo_id}}), HTTP methods, request fields, response fields, and data models.

Feature: {task_list.feature_spec}
Tasks:
{tasks_summary}
"""
    result = structured_llm.invoke(prompt)
    if not isinstance(result, SystemContract):
        raise RuntimeError("Failed to obtain structured SystemContract from Architect Agent LLM response.")
    return result
