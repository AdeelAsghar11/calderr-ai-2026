"""
Fixed FUNCTION_REGISTRY dict mapping function names referenced in YAML to Python callables.
NEVER eval/exec code string from YAML.
"""

from typing import Any, Callable, Dict


def count_words(state: Dict[str, Any]) -> Dict[str, Any]:
    text = state.get("input_text", "")
    words = len(text.split()) if isinstance(text, str) else 0
    return {
        "word_count": words,
        "logs": [f"count_words: counted {words} words"],
    }


def uppercase_transform(state: Dict[str, Any]) -> Dict[str, Any]:
    text = state.get("input_text", "")
    upper = text.upper() if isinstance(text, str) else str(text)
    return {
        "transformed_text": upper,
        "logs": [f"uppercase_transform: transformed input to uppercase"],
    }


def prepare_review_data(state: Dict[str, Any]) -> Dict[str, Any]:
    content = state.get("post_content", "")
    return {
        "review_status": "pending_human_review",
        "logs": [f"prepare_review_data: prepared post content for review ({len(content)} chars)"],
    }


def apply_human_decision(state: Dict[str, Any]) -> Dict[str, Any]:
    decision = state.get("approval_decision", "rejected")
    return {
        "review_status": f"finalized_{decision}",
        "logs": [f"apply_human_decision: recorded decision '{decision}'"],
    }


def check_word_count_quality(state: Dict[str, Any]) -> Dict[str, Any]:
    text = state.get("draft_text", "")
    words = len(text.split()) if isinstance(text, str) else 0
    status = "pass" if words >= 10 else "fail"
    return {
        "word_count": words,
        "quality_status": status,
        "logs": [f"check_word_count_quality: {words} words -> status: {status}"],
    }


def increment_iteration(state: Dict[str, Any]) -> Dict[str, Any]:
    count = state.get("iteration_count", 0)
    new_count = count + 1
    return {
        "iteration_count": new_count,
        "logs": [f"increment_iteration: step {new_count}"],
    }


def handle_support(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "response": "Routed to Customer Support Team.",
        "logs": ["handle_support: dispatched ticket to support queue"],
    }


def handle_sales(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "response": "Routed to Sales & Enterprise Team.",
        "logs": ["handle_sales: dispatched lead to sales representative"],
    }


def handle_technical(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "response": "Routed to Technical Engineering Support.",
        "logs": ["handle_technical: opened engineering issue ticket"],
    }


FUNCTION_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "count_words": count_words,
    "uppercase_transform": uppercase_transform,
    "prepare_review_data": prepare_review_data,
    "apply_human_decision": apply_human_decision,
    "check_word_count_quality": check_word_count_quality,
    "increment_iteration": increment_iteration,
    "handle_support": handle_support,
    "handle_sales": handle_sales,
    "handle_technical": handle_technical,
}
