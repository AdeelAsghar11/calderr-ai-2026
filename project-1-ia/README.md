# Project 1-IA: CalderR AI Assistant (Domain-Specific Chatbot)

This project contains a command-line AI assistant powered by Groq (`llama-3.3-70b-versatile`) and LangChain. It uses domain-specific system prompts to adjust its persona and expertise dynamically, and features interactive commands for enhanced terminal usage.

## Features

- **Domain-Specific Personas:** Change focus topic to specialize in:
  - `programming`: expert assistant for code, architecture, and best practices.
  - `cooking`: culinary expert for recipes, meal plans, and substitutions.
  - `history`: history professor providing historical context.
  - `general`: default concise and helpful assistant.
- **Rich Terminal Interface:** Leverages `rich` for structured panel outputs, color-coded components, Markdown rendering, and tabular help menus.
- **Session Controls:** Interactive slash commands inside the terminal:
  - `/help`: Show command documentation.
  - `/topic`: Interactively switch between domains.
  - `/clear`: Clear the conversation memory history.
  - `/exit`: Exit the chatbot.
- **Token Analytics:** Real-time token usage reporting (input, output, and total tokens) after every assistant turn.

## Prerequisites

Ensure you have your environment set up and the requirements installed. From the workspace root:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # OR if using uv:
   uv pip install -r requirements.txt
   ```
2. Configure your `.env` file in the project root with your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Running the Chatbot

Run the script from the root of the workspace to ensure it loads the `.env` configuration correctly:

```bash
python project-1-ia/chatbot.py
```
