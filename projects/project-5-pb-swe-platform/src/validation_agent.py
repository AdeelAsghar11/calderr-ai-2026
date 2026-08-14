"""
Validation Agent implementation.

Launches a live Docker container from a built image on a dynamic free host port,
exercises stateful HTTP calls against every endpoint in SystemContract, and guarantees container teardown
inside a try...finally block.
"""

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from typing import List, Tuple

from src.schema import DockerBuildResult, EndpointCheck, SystemContract, ValidationReport


def is_docker_available() -> bool:
    """
    Checks if Docker daemon is running and accessible on the current host system.
    """
    try:
        proc = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        return proc.returncode == 0
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


def get_free_port() -> int:
    """
    Binds a socket to port 0 to dynamically allocate an available OS port.
    Prevents port collisions across concurrent test runs or environments.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _is_container_running(container_id: str) -> bool:
    """
    Queries 'docker ps' to verify if a container is currently active.
    """
    try:
        proc = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"id={container_id}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(proc.stdout.strip())
    except Exception:
        return False


def _wait_for_server(base_url: str, max_retries: int = 15, delay: float = 0.5) -> bool:
    """
    Polls the live container HTTP server until it responds or retries expire.
    """
    url = f"{base_url}/todos"
    for _ in range(max_retries):
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status in (200, 201, 404):
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(delay)
    return False


def run_validation_agent(
    contract: SystemContract,
    build_result: DockerBuildResult,
    use_real: bool = False,
) -> ValidationReport:
    """
    Executes live container runtime validation.

    Why teardown is strictly inside a try...finally block:
    If an HTTP check fails or an exception is raised, the finally block guarantees 'docker stop' runs
    so no orphaned containers are leaked on the host system.
    """
    if not build_result.build_succeeded:
        return ValidationReport(
            image_tag=build_result.image_tag,
            container_started=False,
            endpoint_checks=[],
            all_passed=False,
            teardown_succeeded=True,
        )

    host_port = get_free_port()
    base_url = f"http://127.0.0.1:{host_port}"
    container_id = None
    container_started = False
    teardown_succeeded = False
    endpoint_checks: List[EndpointCheck] = []
    created_todo_id = None

    try:
        # Launch detached container
        run_proc = subprocess.run(
            ["docker", "run", "-d", "--rm", "-p", f"{host_port}:8000", build_result.image_tag],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if run_proc.returncode != 0:
            return ValidationReport(
                image_tag=build_result.image_tag,
                container_started=False,
                endpoint_checks=[],
                all_passed=False,
                teardown_succeeded=True,
            )

        container_id = run_proc.stdout.strip()
        container_started = True

        # Wait for server readiness
        ready = _wait_for_server(base_url)
        if not ready:
            endpoint_checks.append(
                EndpointCheck(
                    path="/",
                    method="GET",
                    actual_status=0,
                    passed=False,
                    notes="Container started but live HTTP server failed to respond within readiness timeout",
                )
            )
            return ValidationReport(
                image_tag=build_result.image_tag,
                container_started=True,
                endpoint_checks=endpoint_checks,
                all_passed=False,
                teardown_succeeded=False,
            )

        # Dynamic contract-drift check: Loop over every endpoint in SystemContract
        for ep in contract.endpoints:
            path = ep.path
            method = ep.method

            try:
                if method == "POST" and path == "/todos":
                    payload = json.dumps({"title": "Live Validation Item"}).encode("utf-8")
                    req = urllib.request.Request(
                        f"{base_url}{path}",
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=5.0) as resp:
                        status = resp.status
                        body = json.loads(resp.read().decode("utf-8"))
                        created_todo_id = body.get("id", 1)
                        passed = status in (200, 201) and "id" in body
                        notes = f"Returned status {status}, payload: {body}"

                elif method == "GET" and path == "/todos":
                    req = urllib.request.Request(f"{base_url}{path}", method="GET")
                    with urllib.request.urlopen(req, timeout=5.0) as resp:
                        status = resp.status
                        body = json.loads(resp.read().decode("utf-8"))
                        passed = status == 200
                        notes = f"Returned status {status}, payload: {body}"

                elif method == "PUT" and "/todos/" in path:
                    todo_id = created_todo_id if created_todo_id is not None else 1
                    real_path = path.replace("{todo_id}", str(todo_id))
                    req = urllib.request.Request(f"{base_url}{real_path}", method="PUT")
                    with urllib.request.urlopen(req, timeout=5.0) as resp:
                        status = resp.status
                        body = json.loads(resp.read().decode("utf-8"))
                        passed = status == 200
                        notes = f"Returned status {status}, payload: {body}"

                elif method == "DELETE" and "/todos/" in path:
                    todo_id = created_todo_id if created_todo_id is not None else 1
                    real_path = path.replace("{todo_id}", str(todo_id))
                    req = urllib.request.Request(f"{base_url}{real_path}", method="DELETE")
                    with urllib.request.urlopen(req, timeout=5.0) as resp:
                        status = resp.status
                        body = json.loads(resp.read().decode("utf-8"))
                        passed = status == 200
                        notes = f"Returned status {status}, payload: {body}"

                else:
                    # Generic fallback check for extra endpoints
                    req = urllib.request.Request(f"{base_url}{path}", method=method)
                    with urllib.request.urlopen(req, timeout=5.0) as resp:
                        status = resp.status
                        passed = status in (200, 201, 204)
                        notes = f"Generic endpoint check returned status {status}"

                endpoint_checks.append(
                    EndpointCheck(
                        path=path,
                        method=method,
                        actual_status=status,
                        passed=passed,
                        notes=notes,
                    )
                )
            except urllib.error.HTTPError as e:
                endpoint_checks.append(
                    EndpointCheck(
                        path=path,
                        method=method,
                        actual_status=e.code,
                        passed=False,
                        notes=f"HTTPError: {e.code} {e.reason}",
                    )
                )
            except Exception as ex:
                endpoint_checks.append(
                    EndpointCheck(
                        path=path,
                        method=method,
                        actual_status=0,
                        passed=False,
                        notes=f"Exception during request: {ex}",
                    )
                )

    finally:
        # Teardown guarantee: always stop container
        if container_id:
            try:
                stop_proc = subprocess.run(
                    ["docker", "stop", container_id],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                time.sleep(0.5)
                teardown_succeeded = not _is_container_running(container_id)
            except Exception:
                teardown_succeeded = False
        else:
            teardown_succeeded = True

    all_passed = container_started and len(endpoint_checks) > 0 and all(c.passed for c in endpoint_checks)

    return ValidationReport(
        image_tag=build_result.image_tag,
        container_started=container_started,
        endpoint_checks=endpoint_checks,
        all_passed=all_passed,
        teardown_succeeded=teardown_succeeded,
    )
