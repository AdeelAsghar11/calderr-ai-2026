# Project 4-I-C: Iterative Code Generator

An agentic code generation and validation system built with **LangGraph**, **Python 3.11+**, and **Groq (`llama-3.3-70b-versatile`)**.

The generator iteratively writes Python code, evaluates it inside a sandboxed execution harness against custom test cases, and automatically corrects any errors using feedback loops from failed runs.

---

## 🏗️ Graph Architecture

The workflow forms a self-correcting loop using a `StateGraph`:

```
                      ┌───────────────┐
                      │  generate_fn  │ ◄───────┐
                      └───────┬───────┘         │
                              │                 │
                              ▼                 │
                      ┌───────────────┐         │ (debug)
                      │ execute_node  │         │
                      └───────┬───────┘         │
                              │                 │
               ┌──────────────┴──────────────┐  │
               ▼                             ▼  │
      [passed == True]               [passed == False]
               │                             │
               ▼                             ▼
       ┌──────────────┐             ┌─────────────────┐
       │ success_node │             │   debug_node    │
       └───────┬──────┘             └────────┬────────┘
               │                             │
               ▼                     [iter < max_iter]
              END                            │
                                             └───► (retry)
                                             
                                     [iter >= max_iter]
                                             │
                                             ▼
                                     ┌───────────────┐
                                     │ give_up_node  │
                                     └───────┬───────┘
                                             │
                                             ▼
                                            END
```

### Graph Nodes & Roles
- **`generate`**: Invokes the LLM to write or fix code based on the problem description and any prior error feedback.
- **`execute`**: Executes the generated code in a sandboxed subprocess against strict test cases and captures pass/fail status or error output.
- **`debug`**: Logs formatted failure diagnostic messages and prepares the state for the next generation attempt.
- **`success`**: Marks the problem solved and records the iteration count.
- **`give_up`**: Stops execution if the iteration cap (`MAX_ITERATIONS = 5`) is reached without passing.

---

## 🔒 Sandboxed Subprocess Execution

Each code execution attempt runs inside an isolated temporary directory with multiple layers of sandboxing:

1. **Subprocess Isolation**: Scripts run in dedicated temporary directories created via `tempfile.TemporaryDirectory()`.
2. **Wall-Clock Timeout**: Subprocesses are bounded by a 6.0-second execution cap to kill infinite loops.
3. **Resource Limits (POSIX)**: Enforces memory (`RLIMIT_AS`), CPU time (`RLIMIT_CPU`), file size (`RLIMIT_FSIZE`), and open descriptor (`RLIMIT_NOFILE`) caps on supported Unix environments.
4. **Network Namespace Isolation (Linux)**: Automatically uses `unshare --net` when supported by the host OS to block outbound network calls.
5. **Cross-Platform Compatibility**: Fully supports Windows, Linux, and macOS using `sys.executable` and platform-aware environment isolation.

---

## 🧪 Problem Suite (16 Benchmark Problems)

| Problem ID | Description |
| :--- | :--- |
| `is_palindrome` | Check alphanumeric palindrome ignoring case and punctuation |
| `fizzbuzz` | Standard FizzBuzz sequence |
| `reverse_words` | Reverse word order in space-separated string |
| `is_prime` | Prime number detection |
| `fibonacci` | 0-indexed Fibonacci number calculation |
| `count_vowels` | Count vowels in string |
| `remove_duplicates` | Remove duplicates while preserving original order |
| `binary_search` | Binary search on sorted array |
| `merge_sorted` | Merge two sorted integer lists |
| `most_frequent` | Find element with highest frequency |
| `string_compress` | Consecutive character count compression |
| `eval_rpn` | Reverse Polish Notation evaluation with zero-truncation division |
| `flatten_list` | Arbitrarily nested list flattening (preserving strings) |
| `longest_consecutive` | Length of longest consecutive sequence |
| `snake_to_camel` | Convert `snake_case` to `camelCase` preserving leading/trailing underscores |
| `parse_query_string` | Parse URL query strings into structured dictionaries |

---

## 🚀 Quick Start & Usage

### Prerequisites
- Python 3.11+
- Groq API Key set in `.env` (`GROQ_API_KEY`)

### 1. Offline Smoke Test Suite
Run deterministic offline graph tests:
```bash
uv run python projects/project-4-ic-code-generator/smoke_test.py
```

### 2. Solve a Single Problem
Solve a specific problem using the live Groq backend:
```bash
uv run python projects/project-4-ic-code-generator/code_generator.py solve snake_to_camel
```

### 3. Run Full Benchmark Evaluation Suite
Run all 16 problems in sequence and display solve statistics:
```bash
uv run python projects/project-4-ic-code-generator/code_generator.py run-all
```

---

## 📂 Directory Structure

```text
projects/project-4-ic-code-generator/
├── README.md               # Architecture & documentation
├── code_generator.py       # Core LangGraph iterative code generator
└── smoke_test.py           # Offline verification suite
```

---

## 💻 Tech Stack

- **LLM Engine:** Groq API (`llama-3.3-70b-versatile`)
- **State Machine:** LangGraph (`StateGraph`, `END`)
- **Sandbox Execution:** Python `subprocess`, `tempfile`, `resource`
