FROM python:3.11-slim

WORKDIR /app

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY projects/project-4-pc-workflow-platform ./projects/project-4-pc-workflow-platform

# Install dependencies
RUN uv sync --frozen

EXPOSE 8000 8501

CMD ["uv", "run", "uvicorn", "projects.project-4-pc-workflow-platform.src.api:app", "--host", "0.0.0.0", "--port", "8000"]
