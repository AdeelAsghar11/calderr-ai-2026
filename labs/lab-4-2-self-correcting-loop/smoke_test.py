"""
Smoke test for Tuesday: both the classification router and the
self-correcting loop, using injected fakes instead of real Groq calls.

Run: python smoke_test.py
"""

import os

# Must run before self_correcting_loop is imported: that module's
# FileHandler opens loop_log.txt at import time, so removing the file
# afterward would delete it out from under an already-open handle and
# every write below would silently go to an orphaned, nameless inode.
if os.path.exists("loop_log.txt"):
    os.remove("loop_log.txt")

from importlib import import_module

router = import_module("00_classification_router")
loop = import_module("self_correcting_loop")


# ---------------------------------------------------------------------------
# Router: fake classify/respond, keyword-based, deterministic
# ---------------------------------------------------------------------------
def fake_classify(query: str) -> str:
    q = query.lower()
    technical_kw = ["error", "bug", "docker", "crash", "traceback", "install", "exception", "api"]
    sensitive_kw = ["legal", "lawyer", "rights", "fired", "terminated", "lawsuit", "diagnosed"]
    if any(w in q for w in sensitive_kw):
        return "sensitive"
    if any(w in q for w in technical_kw):
        return "technical"
    return "general"


def fake_respond(query: str, category: str) -> str:
    return f"[{category} handler] would respond to: {query}"


def test_router():
    print("=== Classification router ===")
    graph = router.build_graph(classify_fn=fake_classify, respond_fn=fake_respond)

    cases = [
        ("What's a good recipe for chicken curry?", "general"),
        ("My Docker container keeps crashing with exit code 137, what does that mean?", "technical"),
        ("Can you help me understand my rights if I think I was wrongfully terminated from my job?", "sensitive"),
    ]
    for query, expected in cases:
        result = graph.invoke({"query": query})
        status = "OK" if result["category"] == expected else "MISMATCH"
        print(f"  [{status}] expected={expected:10s} got={result['category']:10s} query={query[:55]!r}")
        assert result["category"] == expected, f"routing failed for: {query}"
    print("  all 3 categories routed correctly\n")


# ---------------------------------------------------------------------------
# Loop: scripted generator, one product per outcome (pass-1 / pass-later / give-up)
# ---------------------------------------------------------------------------
DRAFTS_BY_PRODUCT = {
    "Brew": [
        "Brew: small-batch coffee, delivered weekly to you",  # 7 words, has name -> passes immediately
    ],
    "TrailMate": [
        "TrailMate uses AI to help you plan the perfect hiking route every time",  # 13 words -> fails
        "TrailMate plans your perfect hiking route using AI power",               # 9 words -> fails
        "TrailMate: AI-planned hiking routes, made simple",                        # 6 words, has name -> passes
    ],
    "Ledger": [
        "Track every expense in seconds, built for freelancers",       # 8 words but no "Ledger" -> fails
        "Simple expense tracking made for busy freelancers everywhere",  # 8 words, no "Ledger" -> fails
        "Freelance expense tracking, simplified and fast for you",     # 8 words, no "Ledger" -> fails -> gives up
    ],
}


def make_scripted_generator(drafts_by_product):
    counters = {}

    def scripted_generate(product_name, brief, feedback=None):
        i = counters.get(product_name, 0)
        drafts = drafts_by_product[product_name]
        draft = drafts[min(i, len(drafts) - 1)]
        counters[product_name] = i + 1
        return draft

    return scripted_generate


def test_loop():
    print("=== Self-correcting loop ===")
    graph = loop.build_graph(generate_fn=make_scripted_generator(DRAFTS_BY_PRODUCT))

    # Case A: passes on attempt 1
    r = graph.invoke({"product_name": "Brew", "brief": "small-batch coffee delivered weekly", "max_attempts": 3})
    print(f"  Brew:      gave_up={r['gave_up']:5} iterations={r['iterations_used']}  -> {r['final_response']!r}")
    assert r["gave_up"] is False and r["iterations_used"] == 1

    # Case B: fails twice, passes on attempt 3
    r = graph.invoke({"product_name": "TrailMate", "brief": "AI hiking route planner", "max_attempts": 3})
    print(f"  TrailMate: gave_up={r['gave_up']:5} iterations={r['iterations_used']}  -> {r['final_response']!r}")
    assert r["gave_up"] is False and r["iterations_used"] == 3

    # Case C: fails all 3, gives up gracefully instead of looping forever
    r = graph.invoke({"product_name": "Ledger", "brief": "expense tracking for freelancers", "max_attempts": 3})
    print(f"  Ledger:    gave_up={r['gave_up']:5} iterations={r['iterations_used']}  -> {r['final_response']!r}")
    assert r["gave_up"] is True and r["iterations_used"] == 3

    print("  pass-immediately, pass-after-retry, and give-up-after-max-attempts all verified\n")


if __name__ == "__main__":
    test_router()
    test_loop()

    print("All Tuesday assertions passed.")
    print("\n--- loop_log.txt contents ---")
    with open("loop_log.txt") as f:
        print(f.read())
