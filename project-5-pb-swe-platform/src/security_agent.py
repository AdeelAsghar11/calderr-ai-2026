"""
Security Agent implementation.

Performs static analysis security scanning on generated Backend and Frontend source code artifacts.
Detects OWASP Top 10 vulnerability patterns (SQL Injection, Hardcoded Secrets, Missing Input Validation,
DOM XSS) and produces a structured SecurityReport with classified severity ratings.
"""

import ast
import os
import re
from typing import List

from src.schema import GeneratedComponent, SecurityFinding, SecurityReport


def _scan_backend_code(backend_path: str) -> List[SecurityFinding]:
    """
    Performs static AST and pattern-based security analysis on generated Python backend source code.

    Why static AST and regex analysis is used:
    Static analysis is deterministic, fast, and does not require executing potentially malicious
    or dangerous code payloads.
    """
    findings: List[SecurityFinding] = []
    if not os.path.exists(backend_path):
        return findings

    with open(backend_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Rule 1: SQL Injection Detection (f-strings or string concatenation/formatting in SQL queries)
    sql_fstring_pattern = r'''(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE)\s+.*(\%.*|\.format\(|f["'].*\{.*\}|["']\s*\+)'''
    sql_execute_fstring = r'''execute\s*\(\s*f["'].*\{.*\}'''
    sql_fstring_direct = r'''f["'].*(SELECT|INSERT|UPDATE|DELETE|WHERE).*\{.*\}'''

    if (
        re.search(sql_fstring_pattern, content, re.IGNORECASE)
        or re.search(sql_execute_fstring, content, re.IGNORECASE)
        or re.search(sql_fstring_direct, content, re.IGNORECASE)
    ):
        findings.append(
            SecurityFinding(
                category="SQL Injection",
                severity="critical",
                location=f"{backend_path}: SQL Query Construction",
                description="Unsanitized string interpolation or f-string formatting detected inside SQL query string.",
            )
        )

    # Rule 2: Hardcoded Secrets and Credentials
    secret_pattern = r'''(api_key|secret_key|password|aws_secret|auth_token|jwt_secret)\s*=\s*["'][a-zA-Z0-9_\-\.\=\+]{8,}["']'''
    if re.search(secret_pattern, content, re.IGNORECASE):
        findings.append(
            SecurityFinding(
                category="Hardcoded Credentials",
                severity="high",
                location=f"{backend_path}: Secret Assignment",
                description="Hardcoded API key, secret, or password literal detected in source code.",
            )
        )

    # Rule 3: Missing Input Validation (Untyped route handler parameters in AST)
    try:
        parsed_ast = ast.parse(content)
        for node in ast.walk(parsed_ast):
            if isinstance(node, ast.FunctionDef):
                # Check for route decorators (e.g. @app.get, @app.post)
                has_route_decorator = any(
                    isinstance(d, ast.Call)
                    and isinstance(d.func, ast.Attribute)
                    and d.func.attr in ("get", "post", "put", "delete", "patch")
                    for d in node.decorator_list
                )
                if has_route_decorator:
                    for arg in node.args.args:
                        # If parameter is 'self' or 'cls', ignore
                        if arg.arg in ("self", "cls"):
                            continue
                        if arg.annotation is None:
                            findings.append(
                                SecurityFinding(
                                    category="Input Validation",
                                    severity="medium",
                                    location=f"{backend_path}:{node.name}({arg.arg})",
                                    description=f"Route parameter '{arg.arg}' in handler '{node.name}' lacks type annotations/validation.",
                                )
                            )
    except SyntaxError:
        pass

    return findings


def _scan_frontend_code(frontend_path: str) -> List[SecurityFinding]:
    """
    Performs static pattern-based security analysis on generated HTML/JS frontend source code.

    Scans for client-side XSS patterns such as unescaped DOM assignment.
    """
    findings: List[SecurityFinding] = []
    if not os.path.exists(frontend_path):
        return findings

    with open(frontend_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Rule 4: DOM Cross-Site Scripting (XSS) via unescaped innerHTML or document.write
    inner_html_pattern = r'''\.innerHTML\s*=\s*(?!["']\s*["'])[^;\n]*\+[^;\n]*'''
    doc_write_pattern = r'''document\.write\s*\('''
    eval_pattern = r'''eval\s*\('''

    if re.search(inner_html_pattern, content) or re.search(doc_write_pattern, content) or re.search(eval_pattern, content):
        findings.append(
            SecurityFinding(
                category="Cross-Site Scripting (XSS)",
                severity="medium",
                location=f"{frontend_path}: DOM Assignment",
                description="Unescaped user input or direct dynamic variable assignment into innerHTML / document.write.",
            )
        )

    return findings


def run_security_agent(
    backend_comp: GeneratedComponent,
    frontend_comp: GeneratedComponent,
    use_real: bool = False,
) -> SecurityReport:
    """
    Executes the Security Agent to scan backend and frontend components for security findings.

    Why findings produce severity classifications rather than a binary pass/fail gate:
    Security analysis requires nuance; categorizing findings into Critical, High, Medium, and Low ratings
    enables automated policy gates to differentiate between benign warnings and deployment-blocking flaws.
    """
    if not use_real:
        backend_findings = _scan_backend_code(backend_comp.file_path)
        frontend_findings = _scan_frontend_code(frontend_comp.file_path)
        all_findings = backend_findings + frontend_findings
        critical_count = sum(1 for f in all_findings if f.severity == "critical")

        return SecurityReport(
            findings=all_findings,
            critical_count=critical_count,
        )

    # Real mode execution via Groq structured output
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is required for real mode execution.")

    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=api_key,
    )
    structured_llm = llm.with_structured_output(SecurityReport)

    backend_code = ""
    if os.path.exists(backend_comp.file_path):
        with open(backend_comp.file_path, "r", encoding="utf-8") as f:
            backend_code = f.read()

    frontend_code = ""
    if os.path.exists(frontend_comp.file_path):
        with open(frontend_comp.file_path, "r", encoding="utf-8") as f:
            frontend_code = f.read()

    prompt = f"""You are a Senior Application Security Auditor.
Statically scan the following backend and frontend code artifacts for OWASP Top 10 security vulnerabilities
(including SQL Injection, Hardcoded Credentials, Missing Parameter Input Validation, and Cross-Site Scripting).

Backend Code ({backend_comp.file_path}):
```python
{backend_code}
```

Frontend Code ({frontend_comp.file_path}):
```html
{frontend_code}
```

Produce a structured SecurityReport containing all findings with accurate category, location, description, and severity ('critical', 'high', 'medium', or 'low'). Count total critical findings.
"""
    result = structured_llm.invoke(prompt)
    if not isinstance(result, SecurityReport):
        raise RuntimeError("Failed to obtain structured SecurityReport from Security Agent LLM response.")
    return result
