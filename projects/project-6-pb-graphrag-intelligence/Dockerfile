FROM python:3.11-slim

WORKDIR /app

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy project configuration and workspace files
COPY pyproject.toml uv.lock ./
COPY projects/project-6-pb-graphrag-intelligence ./projects/project-6-pb-graphrag-intelligence

# Install dependencies in non-interactive mode
RUN uv sync --frozen

EXPOSE 8000 8501

CMD ["uv", "run", "uvicorn", "projects.project-6-pb-graphrag-intelligence.api:app", "--host", "0.0.0.0", "--port", "8000"]
