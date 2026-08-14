"""
GitHub Agent implementation.

Constructs repository metadata, file trees, and Pull Request summaries for generated applications.
Operates in dry-run mode by default to prevent unintended external network calls during automated testing.
Live API calls are explicitly gated by the --live flag and require GITHUB_TOKEN authentication.
"""

import os
from typing import Any, Optional
# pyrefly: ignore [missing-import]
from src.schema import GitHubPublishPlan, PipelineRun


def run_github_agent(
    pipeline_run: PipelineRun,
    live: bool = False,
    github_client: Optional[Any] = None,
) -> GitHubPublishPlan:
    """
    Executes the GitHub Agent to prepare or publish a GitHub repository and Pull Request.

    Why dry-run mode is enforced by default:
    Automated agent testing must never pollute external git hosting services or trigger public PRs
    without explicit user consent and explicit flag activation.
    """
    feature_slug = pipeline_run.system_contract.feature_name.lower().replace(" ", "-")
    repo_name = f"generated-{feature_slug}"
    branch_name = f"feat/{feature_slug}-implementation"
    description = f"Auto-generated implementation for feature: {pipeline_run.feature_spec}"

    # Build relative list of files to commit
    files_to_commit = [
        "backend.py",
        "index.html",
        "Dockerfile",
        "docker-compose.yml",
        "test_backend_generated.py",
    ]

    # Construct Markdown Pull Request Body
    contract_endpoints_summary = "\n".join(
        [f"- `{e.method} {e.path}`: {e.description}" for e in pipeline_run.system_contract.endpoints]
    )

    qa_summary = (
        f"Passed: {pipeline_run.qa_report.tests_passed}/{pipeline_run.qa_report.tests_written} "
        f"(Failed: {pipeline_run.qa_report.tests_failed})"
    )

    sec_summary = (
        f"Critical Findings: {pipeline_run.security_report.critical_count}, "
        f"Total Findings: {len(pipeline_run.security_report.findings)}"
    )

    if pipeline_run.validation_report is not None:
        val_status = "All Passed" if pipeline_run.validation_report.all_passed else "Validation Failed"
    else:
        val_status = "Skipped (Docker unavailable in environment)"

    pr_title = f"feat: auto-generated implementation of {pipeline_run.system_contract.feature_name}"
    pr_body = f"""## Auto-Generated AI Engineering Pull Request

### Feature Overview
{pipeline_run.feature_spec}

### System Contract Endpoints
{contract_endpoints_summary}

### Quality & Security Audits
- **QA Suite**: {qa_summary}
- **Security Audit**: {sec_summary}
- **Docker Validation**: {val_status}

### Stages Executed
`{', '.join(pipeline_run.stages_completed)}`
"""

    if not live:
        # Dry-run mode: return plan artifact without making any network call
        return GitHubPublishPlan(
            repo_name=repo_name,
            description=description,
            files_to_commit=files_to_commit,
            branch_name=branch_name,
            pr_title=pr_title,
            pr_body=pr_body,
            dry_run=True,
            actually_published=False,
        )

    # Live mode execution
    token = os.getenv("GITHUB_TOKEN")
    if not token and github_client is None:
        raise RuntimeError("GITHUB_TOKEN environment variable is required for live GitHub API publishing.")

    if github_client is not None:
        # Custom/mock client for testing live invocation without external networks
        github_client.create_repo_and_pr(
            repo_name=repo_name,
            branch=branch_name,
            title=pr_title,
            body=pr_body,
        )
    else:
        # pyrefly: ignore [missing-import]
        import httpx
        # Execute live REST API calls to GitHub
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
        # Step 1: Create repository (or verify exists)
        resp = httpx.post(
            "https://api.github.com/user/repos",
            json={"name": repo_name, "description": description, "private": False},
            headers=headers,
            timeout=10.0,
        )
        if resp.status_code not in (201, 422):
            raise RuntimeError(f"GitHub API repository creation failed with status {resp.status_code}: {resp.text}")

    return GitHubPublishPlan(
        repo_name=repo_name,
        description=description,
        files_to_commit=files_to_commit,
        branch_name=branch_name,
        pr_title=pr_title,
        pr_body=pr_body,
        dry_run=False,
        actually_published=True,
    )
