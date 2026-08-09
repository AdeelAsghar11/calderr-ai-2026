FROM python:3.11-slim

WORKDIR /app

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy project configuration and workspace files
COPY pyproject.toml uv.lock ./
COPY project-6-pb-graphrag-intelligence ./project-6-pb-graphrag-intelligence

# Install dependencies in non-interactive mode
RUN uv sync --frozen

EXPOSE 8000 8501

CMD ["uv", "run", "uvicorn", "project-6-pb-graphrag-intelligence.api:app", "--host", "0.0.0.0", "--port", "8000"]
