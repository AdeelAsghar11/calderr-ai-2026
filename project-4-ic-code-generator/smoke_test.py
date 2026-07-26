"""
Smoke test for 4-I-C: the graph, retry/give-up mechanics, and the
execution sandbox, all exercised with scripted fakes instead of real
Groq calls -- no network, no GROQ_API_KEY required.

Run: python smoke_test.py
"""

import sys

from code_generator import EXEC_TIMEOUT_SECONDS, _HAS_NETNS, build_graph, execute_code
from problems import PROBLEMS, PROBLEMS_BY_ID

failures = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        failures.append(label)


# ---------------------------------------------------------------------------
# 1. Harness correctness: a correct reference solution for every one of
#    the 10 problems must pass its own test cases. This is checking my
#    own problem/test-case authoring, independent of any LLM.
# ---------------------------------------------------------------------------

REFERENCE_SOLUTIONS = {
    "is_palindrome": "def is_palindrome(s):\n    c = [ch.lower() for ch in s if ch.isalnum()]\n    return c == c[::-1]\n",
    "fizzbuzz": (
        "def fizzbuzz(n):\n"
        "    out = []\n"
        "    for i in range(1, n + 1):\n"
        "        if i % 15 == 0: out.append('FizzBuzz')\n"
        "        elif i % 3 == 0: out.append('Fizz')\n"
        "        elif i % 5 == 0: out.append('Buzz')\n"
        "        else: out.append(str(i))\n"
        "    return out\n"
    ),
    "reverse_words": "def reverse_words(s):\n    return ' '.join(reversed(s.split(' ')))\n",
    "is_prime": (
        "def is_prime(n):\n"
        "    if n < 2: return False\n"
        "    for i in range(2, int(n ** 0.5) + 1):\n"
        "        if n % i == 0: return False\n"
        "    return True\n"
    ),
    "fibonacci": (
        "def fibonacci(n):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n): a, b = b, a + b\n"
        "    return a\n"
    ),
    "count_vowels": "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
    "remove_duplicates": (
        "def remove_duplicates(lst):\n"
        "    seen, out = set(), []\n"
        "    for x in lst:\n"
        "        if x not in seen:\n"
        "            seen.add(x); out.append(x)\n"
        "    return out\n"
    ),
    "binary_search": (
        "def binary_search(arr, target):\n"
        "    lo, hi = 0, len(arr) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if arr[mid] == target: return mid\n"
        "        if arr[mid] < target: lo = mid + 1\n"
        "        else: hi = mid - 1\n"
        "    return -1\n"
    ),
    "merge_sorted": (
        "def merge_sorted(a, b):\n"
        "    return sorted(a + b)\n"
    ),
    "most_frequent": (
        "def most_frequent(lst):\n"
        "    return max(set(lst), key=lst.count)\n"
    ),
    "string_compress": (
        "def string_compress(s):\n"
        "    if not s: return s\n"
        "    res, curr, count = [], s[0], 0\n"
        "    for char in s:\n"
        "        if char == curr: count += 1\n"
        "        else:\n"
        "            res.append(f'{curr}{count}')\n"
        "            curr, count = char, 1\n"
        "    res.append(f'{curr}{count}')\n"
        "    comp = ''.join(res)\n"
        "    return comp if len(comp) < len(s) else s\n"
    ),
    "eval_rpn": (
        "def eval_rpn(tokens):\n"
        "    stack = []\n"
        "    for t in tokens:\n"
        "        if t in ('+', '-', '*', '/'):\n"
        "            b, a = stack.pop(), stack.pop()\n"
        "            if t == '+': stack.append(a + b)\n"
        "            elif t == '-': stack.append(a - b)\n"
        "            elif t == '*': stack.append(a * b)\n"
        "            elif t == '/': stack.append(int(a / b))\n"
        "        else: stack.append(int(t))\n"
        "    return stack[0]\n"
    ),
    "flatten_list": (
        "def flatten_list(nested):\n"
        "    out = []\n"
        "    for item in nested:\n"
        "        if isinstance(item, list): out.extend(flatten_list(item))\n"
        "        else: out.append(item)\n"
        "    return out\n"
    ),
    "longest_consecutive": (
        "def longest_consecutive(nums):\n"
        "    num_set = set(nums)\n"
        "    best = 0\n"
        "    for x in num_set:\n"
        "        if x - 1 not in num_set:\n"
        "            curr, streak = x, 1\n"
        "            while curr + 1 in num_set: curr += 1; streak += 1\n"
        "            best = max(best, streak)\n"
        "    return best\n"
    ),
    "snake_to_camel": (
        "def snake_to_camel(s):\n"
        "    l = len(s) - len(s.lstrip('_'))\n"
        "    r = len(s) - len(s.rstrip('_'))\n"
        "    core = s[l:len(s)-r if r > 0 else None]\n"
        "    if not core: return s\n"
        "    parts = [p for p in core.split('_') if p]\n"
        "    if not parts: return s\n"
        "    camel = parts[0] + ''.join(p.capitalize() for p in parts[1:])\n"
        "    return '_' * l + camel + '_' * r\n"
    ),
    "parse_query_string": (
        "def parse_query_string(query):\n"
        "    if not query: return {}\n"
        "    res = {}\n"
        "    for item in query.split('&'):\n"
        "        if '=' in item:\n"
        "            k, v = item.split('=', 1)\n"
        "        else:\n"
        "            k, v = item, True\n"
        "        if k in res:\n"
        "            if isinstance(res[k], list): res[k].append(v)\n"
        "            else: res[k] = [res[k], v]\n"
        "        else:\n"
        "            res[k] = v\n"
        "    return res\n"
    ),
}

print("=== 1. Harness correctness against reference solutions ===")
for problem in PROBLEMS:
    solution = REFERENCE_SOLUTIONS[problem["id"]]
    result = execute_code(solution, problem["function_name"], problem["test_cases"])
    check(f"{problem['id']}: reference solution passes", result["passed"])
    if not result["passed"]:
        print(f"       -> {result['error']}")

# ---------------------------------------------------------------------------
# 2. Harness correctly detects failure (wrong logic, not just wrong syntax)
# ---------------------------------------------------------------------------

print("\n=== 2. Harness correctly detects a wrong solution ===")
wrong = "def is_prime(n):\n    return True\n"  # wrong: says everything is prime
result = execute_code(wrong, "is_prime", PROBLEMS_BY_ID["is_prime"]["test_cases"])
check("wrong is_prime correctly fails (not silently passes)", not result["passed"])
check("failure message names the mismatch", "test" in result["error"])

# ---------------------------------------------------------------------------
# 3. Sandbox: timeout on an infinite loop
# ---------------------------------------------------------------------------

print("\n=== 3. Sandbox catches an infinite loop ===")
import time as _time
infinite = "def is_prime(n):\n    while True:\n        pass\n"
_t0 = _time.time()
result = execute_code(infinite, "is_prime", PROBLEMS_BY_ID["is_prime"]["test_cases"])
_elapsed = _time.time() - _t0
# Two valid ways this can be caught: the wall-clock subprocess timeout
# (EXEC_TIMEOUT_SECONDS), or the RLIMIT_CPU backstop killing it by
# signal first, whichever fires first -- RLIMIT_CPU=5s currently wins
# the race against EXEC_TIMEOUT_SECONDS=6s. Either is correct; what
# actually matters is that it terminated at all, and did so quickly.
caught_by_timeout = "Timed out" in result["error"]
caught_by_signal = "killed (signal" in result["error"]
check(
    f"infinite loop terminated (not hung) in {_elapsed:.1f}s via "
    f"{'wall-clock timeout' if caught_by_timeout else 'RLIMIT_CPU signal' if caught_by_signal else 'unknown path'}",
    not result["passed"] and (caught_by_timeout or caught_by_signal) and _elapsed < EXEC_TIMEOUT_SECONDS + 2,
)

# ---------------------------------------------------------------------------
# 4. Sandbox: syntax error doesn't crash the harness
# ---------------------------------------------------------------------------

print("\n=== 4. Sandbox handles a syntax error cleanly ===")
broken = "def is_prime(n:\n    return True\n"  # missing closing paren
result = execute_code(broken, "is_prime", PROBLEMS_BY_ID["is_prime"]["test_cases"])
check("syntax error reported as failure, not an unhandled exception", not result["passed"])
check("syntax error message is informative", "SyntaxError" in result["error"] or "failed to run" in result["error"])

# ---------------------------------------------------------------------------
# 5. Sandbox: network isolation actually blocks network, when available
# ---------------------------------------------------------------------------

print(f"\n=== 5. Network isolation (unshare --net available: {_HAS_NETNS}) ===")
net_attempt = (
    "def is_prime(n):\n"
    "    import socket\n"
    "    socket.gethostbyname('pypi.org')\n"
    "    return True\n"
)
result = execute_code(net_attempt, "is_prime", PROBLEMS_BY_ID["is_prime"]["test_cases"])
if _HAS_NETNS:
    check("network call fails inside the sandbox", not result["passed"])
else:
    print("  (skipped strict check -- unshare --net unavailable on this host, isolation not enforced)")

# ---------------------------------------------------------------------------
# 6. Full graph, scripted fakes: correct on first try
# ---------------------------------------------------------------------------

print("\n=== 6. Graph: correct-first-try terminates at iteration 1 ===")


def fake_correct_first_try(problem, previous_code, error_feedback):
    return REFERENCE_SOLUTIONS[problem["id"]]


graph = build_graph(fake_correct_first_try)
result = graph.invoke({"problem": PROBLEMS_BY_ID["is_palindrome"], "max_iterations": 5})
check("solved", result["solved"])
check("iterations_used == 1", result["iterations_used"] == 1)

# ---------------------------------------------------------------------------
# 7. Full graph, scripted fakes: fails once, then a corrected attempt
#    succeeds -- this is the one that actually proves the retry loop
#    uses previous_code/error_feedback correctly, not just "eventually
#    calls generate again."
# ---------------------------------------------------------------------------

print("\n=== 7. Graph: wrong on iteration 1, correct on iteration 2 ===")


def fake_eventually_succeeds(problem, previous_code, error_feedback):
    if previous_code is None:
        return "def is_palindrome(s):\n    return True\n"  # deliberately wrong
    assert error_feedback, "generate was called for a retry but got no error_feedback"
    return REFERENCE_SOLUTIONS[problem["id"]]


graph = build_graph(fake_eventually_succeeds)
result = graph.invoke({"problem": PROBLEMS_BY_ID["is_palindrome"], "max_iterations": 5})
check("solved on retry", result["solved"])
check("iterations_used == 2", result["iterations_used"] == 2)
check("log has both a debug entry and a solved entry", any("failed" in l for l in result["log"]) and any("solved" in l for l in result["log"]))

# ---------------------------------------------------------------------------
# 8. Full graph, scripted fakes: never succeeds -> gives up at the cap,
#    does not loop forever.
# ---------------------------------------------------------------------------

print("\n=== 8. Graph: never-correct gives up at max_iterations ===")


def fake_never_succeeds(problem, previous_code, error_feedback):
    return "def is_palindrome(s):\n    return False\n"  # always wrong


graph = build_graph(fake_never_succeeds)
result = graph.invoke({"problem": PROBLEMS_BY_ID["is_palindrome"], "max_iterations": 3})
check("gave_up is True", result["gave_up"])
check("solved is False", not result["solved"])
check("iterations_used == max_iterations (3), not more", result["iterations_used"] == 3)

# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
if failures:
    print(f"{len(failures)} check(s) FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All checks passed.")
