# Project 1-I-A: Intelligent CLI Assistant

A terminal-based AI assistant built with **Python 3.11+**, **LangChain**, **Groq (`llama-3.3-70b-versatile`)**, and **Rich**. 

The assistant dynamically adjusts its system prompt persona based on domain topics (`programming`, `cooking`, `history`, `general`), tracks token consumption, and preserves multi-turn conversation memory.

---

## 🏗️ Architecture

```
User Input → Command Parser → SystemMessage (domain prompt)
                                      ↓
                          LangChain ChatGroq (llama-3.3-70b)
                                      ↓
                          Conversation History (in-memory)
                                      ↓
                          Rich Console Output + Token Usage
```

---

## ✨ Features

- **Domain-Specific Personas:** Interactively switch focus between:
  - `programming`: Software engineering expert for code, architecture, and debugging.
  - `cooking`: Culinary consultant for recipes, dietary substitutions, and techniques.
  - `history`: Historian offering detailed historical context and timelines.
  - `general`: Concise, helpful default assistant persona.
- **Rich Terminal Interface:** Structured panel outputs, color-coded components, rendered Markdown, and command summary tables.
- **Session & Memory Control Commands:**
  - `/help`: Displays available slash commands.
  - `/topic`: Switches active domain mode and resets conversation state.
  - `/clear`: Resets active conversation context history.
  - `/exit`: Terminates session safely.
- **Token Analytics:** Reports prompt, completion, and total token usage after each assistant response.

---

## 🚀 Quick Start & Usage

### Prerequisites
- Python 3.11+
- Groq API Key set in `.env` (`GROQ_API_KEY`)

### Run Chatbot
From the repository root:
```bash
uv run python projects/project-1-ia/chatbot.py
```

---

## 📂 Directory Structure

```text
projects/project-1-ia/
├── README.md                 # Project architecture & user guide
└── chatbot.py                # Core CLI assistant & Rich interface
```

---

## 💻 Tech Stack

- **LLM Engine:** Groq API (`llama-3.3-70b-versatile`)
- **Framework:** LangChain & LangChain-Groq
- **Terminal UI:** Rich (`Console`, `Panel`, `Markdown`, `Table`)
