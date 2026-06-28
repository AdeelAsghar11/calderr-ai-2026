# Project 1-PC: Agentic Research Assistant

An advanced, multi-step research agent powered by Streamlit, LangChain, and Groq's `llama-3.3-70b-versatile` model. It takes any user-provided research question, dynamically plans subtopics to explore, researches them sequentially using structured outputs, and synthesizes a professional markdown report.

## Features

- **Dynamic Planning:** Generates a structured research plan consisting of 3 to 5 targeted, non-overlapping subtopics to systematically cover the main question.
- **Sequential Research Loop:** Research agent analyzes each subtopic individually, collecting key facts and calculating a confidence score based on the clarity and validation level of the findings.
- **Automated Synthesis:** Compiles all findings into a final executive summary, mapping out key takeaways, overall research confidence, and limitations.
- **Interactive UI (Streamlit):**
  - Interactive inputs and progress status updates.
  - Interactive confidence charts comparing subtopic confidence levels.
  - Side-by-side executive summary and metrics view.
- **Markdown Export:** Provides a downloadable markdown file containing the formatted report with one click.

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

## Running the Research Assistant

Run the Streamlit application from the root of the workspace:

```bash
streamlit run project-1-pc/research_assistant.py
```
