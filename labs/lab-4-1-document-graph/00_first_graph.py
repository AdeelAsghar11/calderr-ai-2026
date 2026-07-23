"""
Monday warm-up: the smallest possible LangGraph.

Goal: see the State -> Node -> Edge mental model with nothing else in the
way. No LLM, no conditional routing, no persistence -- just three plain
functions wired into a graph and run start to finish.

Run: python 00_first_graph.py
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. STATE -- the shared data structure that flows through every node.
#    TypedDict is just a dict with a type-checked shape. Every node reads
#    from this and returns a partial update to it.
# ---------------------------------------------------------------------------
class GreetingState(TypedDict):
    name: str
    greeting: str
    shout: str


# ---------------------------------------------------------------------------
# 2. NODES -- plain Python functions. Each takes the current state and
#    returns a dict of the keys it wants to change. LangGraph merges that
#    dict into the running state (a shallow update, by default: whatever
#    key you return overwrites that key -- more on this Wednesday when we
#    cover Annotated reducers for keys that should append instead).
# ---------------------------------------------------------------------------
def add_greeting(state: GreetingState) -> dict:
    return {"greeting": f"Hello, {state['name']}!"}


def add_shout(state: GreetingState) -> dict:
    return {"shout": state["greeting"].upper()}


def finalize(state: GreetingState) -> dict:
    print(f"[finalize node saw]: {state['shout']}")
    return {}


# ---------------------------------------------------------------------------
# 3. EDGES -- plain, unconditional edges. Each one just says "after this
#    node, run that node next." START and END are LangGraph's built-in
#    sentinel nodes marking where a run begins and where it must stop.
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(GreetingState)

    builder.add_node("greet", add_greeting)
    builder.add_node("shout", add_shout)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "greet")
    builder.add_edge("greet", "shout")
    builder.add_edge("shout", "finalize")
    builder.add_edge("finalize", END)

    # .compile() validates the graph structure (e.g. catches edges that
    # point at nodes you never added) and returns a runnable object. A
    # compiled graph is a Runnable, so it has .invoke(), .stream(), etc.
    # exactly like any LangChain chain -- LangGraph is not a separate
    # execution model bolted onto LangChain, it produces the same kind of
    # object your LCEL chains already are.
    return builder.compile()


if __name__ == "__main__":
    app = build_graph()

    # A quick look at the graph's shape before running it -- useful any
    # time you want to sanity-check the wiring without executing anything.
    print("Graph structure:")
    print(app.get_graph().draw_ascii())

    result = app.invoke({"name": "Adeel"})
    print("\nFinal state returned by invoke():")
    print(result)
