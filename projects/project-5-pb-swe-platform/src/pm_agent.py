"""
PM (Product Manager) Agent implementation.

Decomposes raw natural language feature specifications into structured TaskList objects containing
discrete engineering tasks and testable acceptance criteria. Supports deterministic offline stub execution
and live LLM structured output via ChatGroq.
"""

import os
# pyrefly: ignore [missing-import]
from src.schema import Task, TaskList


def run_pm_agent(feature_spec: str, use_real: bool = False) -> TaskList:
    """
    Executes the PM Agent to decompose a feature specification into a structured TaskList.

    Why stub mode is deterministic:
    Offline unit tests and CI pipelines require fast, deterministic outputs that do not depend
    on external LLM API availability or floating tokens.

    Why real mode uses ChatGroq with structured output:
    When real mode is toggled, structured output guarantees Pydantic validation before downstream
    agents consume the TaskList.
    """
    if not use_real:
        # Stub implementation providing fixed, correct task decomposition for the standard todo spec
        return TaskList(
            feature_spec=feature_spec,
            tasks=[
                Task(
                    description="Implement POST /todos endpoint to add a new todo item",
                    acceptance_criteria=[
                        "Accepts a title in JSON payload",
                        "Returns 201 Created with auto-generated id and default completed=False",
                    ],
                ),
                Task(
                    description="Implement GET /todos endpoint to list all todo items",
                    acceptance_criteria=[
                        "Returns JSON payload containing array of all todo items",
                        "Includes id, title, and completed status for each item",
                    ],
                ),
                Task(
                    description="Implement PUT /todos/{todo_id}/complete endpoint to mark todo complete",
                    acceptance_criteria=[
                        "Updates target todo completed flag to True",
                        "Returns updated todo item or 404 error if id not found",
                    ],
                ),
                Task(
                    description="Implement DELETE /todos/{todo_id} endpoint to delete a todo",
                    acceptance_criteria=[
                        "Removes target todo from storage",
                        "Returns confirmation message",
                    ],
                ),
            ],
        )

    # Real mode execution via Groq
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
    structured_llm = llm.with_structured_output(TaskList)

    prompt = f"""You are a Senior Technical Product Manager.
Decompose the following feature spec into concrete engineering tasks with explicit acceptance criteria.

Feature Spec:
{feature_spec}
"""
    result = structured_llm.invoke(prompt)
    if not isinstance(result, TaskList):
        raise RuntimeError("Failed to obtain structured TaskList from PM Agent LLM response.")
    return result
