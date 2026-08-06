"""
DevOps Agent implementation.

Writes Dockerfile and docker-compose.yml configuration files matching repository conventions (python:3.11-slim).
Executes 'docker build' as a real subprocess, tracking duration, build success status, and log output.
"""

import os
import subprocess
import time
from src.schema import DockerBuildResult, GeneratedComponent


def run_devops_agent(
    backend_comp: GeneratedComponent,
    output_dir: str,
    image_tag: str = "project5_backend:latest",
    use_real: bool = False,
) -> DockerBuildResult:
    """
    Executes the DevOps Agent to build a Docker image for the generated backend application.

    Why Dockerfile uses python:3.11-slim:
    Matching the repo reference (project-4-pc-workflow-platform/Dockerfile) ensures consistent
    container runtime environments across all production-tier projects.
    """
    os.makedirs(output_dir, exist_ok=True)
    dockerfile_path = os.path.abspath(os.path.join(output_dir, "Dockerfile"))
    compose_path = os.path.abspath(os.path.join(output_dir, "docker-compose.yml"))

    if not use_real:
        # Standard production Dockerfile template
        dockerfile_content = '''FROM python:3.11-slim

WORKDIR /app

# Install FastAPI and Uvicorn dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic

# Copy generated backend source code
COPY backend.py ./

EXPOSE 8000

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        compose_content = f'''version: '3.8'

services:
  backend:
    build: .
    image: {image_tag}
    ports:
      - "8000:8000"
'''
    else:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is required for real mode execution.")

        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=api_key,
        )

        prompt = f"""You are a Lead DevOps / Infrastructure Engineer.
Generate a valid Dockerfile for a FastAPI python application file located at '{backend_comp.file_path}'.
Base image MUST be 'python:3.11-slim'. Install fastapi and uvicorn. Expose port 8000.
Return ONLY raw Dockerfile content inside a ```dockerfile ``` block.
"""
        response = llm.invoke(prompt)
        raw_text = str(response.content)

        if "```dockerfile" in raw_text:
            dockerfile_content = raw_text.split("```dockerfile")[1].split("```")[0].strip()
        elif "```" in raw_text:
            dockerfile_content = raw_text.split("```")[1].split("```")[0].strip()
        else:
            dockerfile_content = raw_text.strip()

        compose_content = f'''version: '3.8'

services:
  backend:
    build: .
    image: {image_tag}
    ports:
      - "8000:8000"
'''

    with open(dockerfile_path, "w", encoding="utf-8") as f:
        f.write(dockerfile_content)

    with open(compose_path, "w", encoding="utf-8") as f:
        f.write(compose_content)

    # Execute docker build in a real subprocess
    start_time = time.time()
    try:
        proc = subprocess.run(
            ["docker", "build", "-t", image_tag, "."],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=300.0,
        )
        duration = round(time.time() - start_time, 2)
        build_succeeded = proc.returncode == 0
        build_output = proc.stdout + "\n" + proc.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        duration = round(time.time() - start_time, 2)
        build_succeeded = False
        build_output = f"Docker build execution failed or timed out: {e}"

    return DockerBuildResult(
        image_tag=image_tag,
        build_succeeded=build_succeeded,
        build_output=build_output,
        build_duration_seconds=duration,
    )
