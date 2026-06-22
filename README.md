# CalderR Agentic AI Engineering Internship 2026

**Name:** Adeel Asghar  
**GitHub:** AdeelAsghar11  
**Programme:** CalderR Agentic AI Engineering Internship — Week 1  

---

## Project 1-I-A: Intelligent CLI Assistant

A terminal-based AI assistant powered by Groq and LangChain with conversation memory, domain switching, and Rich terminal UI.

### Architecture

```
User Input → Command Parser → SystemMessage (domain prompt)
                                      ↓
                          LangChain ChatGroq (llama-3.3-70b)
                                      ↓
                          Conversation History (in-memory)
                                      ↓
                          Rich Console Output + Token Usage
```

### Features

- Multi-turn conversation with persistent history
- 4 domain modes: `programming`, `cooking`, `history`, `general`
- Token usage displayed after every response
- `/clear`, `/topic`, `/help`, `/exit` commands
- Graceful error handling — failed calls don't corrupt history

### Setup

```bash
# Clone the repo
git clone git@github.com:AdeelAsghar11/calderr-ai-2026.git
cd calderr-ai-2026

# Install dependencies
uv sync

# Add your API key
cp env.example .env
# Edit .env and add your GROQ_API_KEY

# Run
uv run python chatbot.py
```

### Commands

| Command  | Description                        |
|----------|------------------------------------|
| `/help`  | Show available commands            |
| `/clear` | Clear conversation history         |
| `/topic` | Switch domain (clears history)     |
| `/exit`  | Quit                               |

### Example Conversations

**1. General domain — context retention**
```
You: what is a neural network?
Assistant: A neural network is a computational model inspired by the brain...
Tokens — in: 57 · out: 210 · total: 267

You: can you give me a Python example of one?
Assistant: Sure, building on what I explained above... [uses previous context]
Tokens — in: 280 · out: 310 · total: 590
```

**2. Programming domain — domain guard**
```
You: /topic → programming

You: how do I reverse a list in Python?
Assistant: You can reverse a list using slice notation: my_list[::-1]...

You: what's a good pasta recipe?
Assistant: I'm a programming assistant and can't help with cooking questions.
Please ask a programming-related question.
```

**3. History domain — multi-turn**
```
You: /topic → history

You: who was julius caesar?
Assistant: Julius Caesar was a Roman general and statesman...
Tokens — in: 75 · out: 280 · total: 355

You: what led to his assassination?
Assistant: Building on his rise to power I described... [retains context]
Tokens — in: 368 · out: 245 · total: 613
```

### Skills Demonstrated

- LangChain `ChatGroq` with `SystemMessage`, `HumanMessage`, `AIMessage`
- Manual conversation history management
- Domain-based persona switching
- Rich terminal UI (`Panel`, `Markdown`, `Table`, `Console`)
- Token usage tracking via `response.usage_metadata`
- Type hints throughout (`dict[str, str]`, `list[HumanMessage | AIMessage]`)
- Error handling that preserves conversation state

### Stack

- Python 3.11+
- LangChain + LangChain-Groq
- Groq API (`llama-3.3-70b-versatile`)
- Rich
- python-dotenv
- uv