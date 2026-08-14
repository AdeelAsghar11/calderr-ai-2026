"""
Phase 2 Pytest Verification Suite.

Tests QA and Security quality gate agents across 4 required test cases:
1. QA Happy Path: QA Agent on Phase 1's clean stub backend -> tests_passed == tests_written.
2. QA Failure Detection Proof: QA Agent on deliberately broken backend fixture -> at least one test passed=False.
3. Security Detection Proof: Security Agent on vulnerable backend fixture (SQL injection) -> flags severity high/critical and names injection category.
4. False-Positive Check: Security Agent on Phase 1's clean backend -> zero critical findings.
"""

import os
import sys
# pyrefly: ignore [missing-import]
import pytest

# Insert project directory to sys.path so src imports resolve cleanly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# pyrefly: ignore [missing-import]
from src.pipeline import run_phase1_pipeline, run_phase2_pipeline
# pyrefly: ignore [missing-import]
from src.pm_agent import run_pm_agent
# pyrefly: ignore [missing-import]
from src.qa_agent import run_qa_agent
# pyrefly: ignore [missing-import]
from src.security_agent import run_security_agent
# pyrefly: ignore [missing-import]
from src.schema import GeneratedComponent, QAReport, SecurityReport

FIXED_TODO_SPEC = (
    "A simple todo list API: users can add a todo, list all todos, "
    "mark a todo complete, and delete a todo."
)


def test_qa_happy_path(tmp_path):
    """
    Case 1: QA Happy Path.
    Run QA Agent against Phase 1's clean, stub-generated backend.
    Assert tests_written matches criteria count and tests_passed == tests_written.
    """
    output_dir = str(tmp_path / "qa_happy_path")
    pipeline_result = run_phase2_pipeline(
        feature_spec=FIXED_TODO_SPEC,
        output_dir=output_dir,
        use_real=False,
    )

    qa_report: QAReport = pipeline_result["qa"]
    assert qa_report.tests_written > 0, "QA Agent did not write any test functions"
    assert qa_report.tests_passed == qa_report.tests_written, (
        f"Expected all {qa_report.tests_written} tests to pass, but only {qa_report.tests_passed} passed."
    )
    assert qa_report.tests_failed == 0, f"Expected 0 failed tests, got {qa_report.tests_failed}"


def test_qa_failure_detection_proof(tmp_path):
    """
    Case 2: QA Failure Detection Proof.
    Execute QA Agent against a deliberately broken backend fixture file.
    Assert at least one test is reported as passed=False.
    """
    # Write a deliberately broken backend fixture file
    broken_backend_path = os.path.join(tmp_path, "broken_backend.py")
    broken_code = '''"""
Deliberately broken backend fixture.
Returns wrong HTTP status codes and missing fields to trigger test failures.
"""
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Broken API")

@app.post("/todos")
def add_todo():
    # Fails criteria expecting 201/200 status and title field
    raise HTTPException(status_code=500, detail="Internal server crash")

@app.get("/todos")
def list_todos():
    # Returns invalid data format
    return "Not a JSON dictionary or list"

@app.put("/todos/{todo_id}/complete")
def complete_todo(todo_id: int):
    raise HTTPException(status_code=404, detail="Missing complete handler")

@app.delete("/todos/{todo_id}")
def delete_todo(todo_id: int):
    raise HTTPException(status_code=404, detail="Missing delete handler")
'''
    with open(broken_backend_path, "w", encoding="utf-8") as f:
        f.write(broken_code)

    broken_comp = GeneratedComponent(
        component_name="backend",
        file_path=broken_backend_path,
        summary="Deliberately broken backend fixture",
    )

    task_list = run_pm_agent(feature_spec=FIXED_TODO_SPEC, use_real=False)
    test_output_dir = str(tmp_path / "broken_qa_run")

    qa_report = run_qa_agent(
        task_list=task_list,
        backend_comp=broken_comp,
        output_dir=test_output_dir,
        use_real=False,
    )

    assert qa_report.tests_failed > 0, "QA Agent failed to detect errors in broken backend fixture!"
    failed_results = [r for r in qa_report.results if not r.passed]
    assert len(failed_results) > 0, "Expected at least one TestResult object with passed=False"


def test_security_detection_proof(tmp_path):
    """
    Case 3: Security Detection Proof.
    Scan a backend fixture containing an explicit SQL Injection flaw.
    Assert Security Agent flags finding with severity 'high' or 'critical' and category naming injection.
    """
    vulnerable_backend_path = os.path.join(tmp_path, "vulnerable_backend.py")
    vulnerable_code = '''"""
Vulnerable backend fixture containing explicit SQL Injection flaw.
"""
from fastapi import FastAPI
import sqlite3

app = FastAPI(title="Vulnerable API")

@app.get("/search")
def search_todos(query: str):
    # Explicit SQL Injection flaw: raw string formatting into SQL query
    sql_query = f"SELECT * FROM todos WHERE title = '{query}'"
    conn = sqlite3.connect("todos.db")
    cursor = conn.cursor()
    cursor.execute(sql_query)
    return cursor.fetchall()
'''
    with open(vulnerable_backend_path, "w", encoding="utf-8") as f:
        f.write(vulnerable_code)

    vulnerable_backend_comp = GeneratedComponent(
        component_name="backend",
        file_path=vulnerable_backend_path,
        summary="Vulnerable backend fixture with SQLi",
    )

    clean_frontend_path = os.path.join(tmp_path, "index.html")
    with open(clean_frontend_path, "w", encoding="utf-8") as f:
        f.write("<html><body>Clean Frontend</body></html>")

    clean_frontend_comp = GeneratedComponent(
        component_name="frontend",
        file_path=clean_frontend_path,
        summary="Clean frontend fixture",
    )

    sec_report: SecurityReport = run_security_agent(
        backend_comp=vulnerable_backend_comp,
        frontend_comp=clean_frontend_comp,
        use_real=False,
    )

    assert len(sec_report.findings) > 0, "Security Agent failed to flag vulnerable backend fixture!"

    sqli_findings = [
        f for f in sec_report.findings
        if f.severity in ("critical", "high") and ("injection" in f.category.lower() or "sql" in f.category.lower())
    ]
    assert len(sqli_findings) > 0, (
        f"Expected high/critical SQL Injection finding, but got: {sec_report.findings}"
    )


def test_security_false_positive_check(tmp_path):
    """
    Case 4: False-Positive Check.
    Run Security Agent against Phase 1's clean generated backend.
    Assert zero 'critical' findings.
    """
    output_dir = str(tmp_path / "sec_clean_run")
    pipeline_result = run_phase2_pipeline(
        feature_spec=FIXED_TODO_SPEC,
        output_dir=output_dir,
        use_real=False,
    )

    sec_report: SecurityReport = pipeline_result["security"]
    assert sec_report.critical_count == 0, f"Expected 0 critical findings on clean backend, got {sec_report.critical_count}"
    critical_findings = [f for f in sec_report.findings if f.severity == "critical"]
    assert len(critical_findings) == 0, f"False positive critical findings detected: {critical_findings}"
