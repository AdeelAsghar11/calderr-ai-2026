"""
fetch_readmes.py — Pull READMEs from your GitHub repos automatically

Uses the GitHub REST API (no auth needed for public repos, 60 requests/hour
unauthenticated — plenty for pulling ~10-20 repos once).

What it does:
  1. Lists your public repos (or a specific list you provide)
  2. Fetches each repo's README via the GitHub API
  3. Lightly cleans it — strips HTML badges, shields.io images, capsule-render
     banners, and excess blank lines (the same kind of cleanup done manually
     for the portfolio and profile README)
  4. Saves each as docs/{repo_name}_readme.md

No dependencies beyond the Python standard library.

Usage
-----
  # Pull ALL public repos
  python fetch_readmes.py

  # Pull only specific repos (recommended — skips forks/trivial repos)
  python fetch_readmes.py --repos BSL-Hand_Gesture_Recognition,ThreatLens,dermvision,sortd,Cifar-10-Classification-using-ML

  # Different GitHub username
  python fetch_readmes.py --user someoneelse
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

GITHUB_USERNAME = "AdeelAsghar11"
OUTPUT_DIR      = Path("docs")
API_BASE        = "https://api.github.com"


def api_get(url: str) -> dict | list | None:
    """GET request to the GitHub API. Returns parsed JSON or None on failure."""
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"  [error] {url} -> HTTP {e.code}")
        return None
    except Exception as e:
        print(f"  [error] {url} -> {e}")
        return None


def list_repos(username: str) -> list[str]:
    """Return all public, non-fork repo names for a user."""
    repos = []
    page  = 1
    while True:
        url  = f"{API_BASE}/users/{username}/repos?per_page=100&page={page}&type=owner"
        data = api_get(url)
        if not data:
            break
        for repo in data:
            if not repo.get("fork", False):
                repos.append(repo["name"])
        if len(data) < 100:
            break
        page += 1
    return repos


def fetch_readme_raw(username: str, repo: str) -> str | None:
    """
    Fetch a repo's README as raw markdown text.
    Uses the special 'raw' Accept header so GitHub returns plain text
    instead of a base64-encoded JSON blob.
    """
    url = f"{API_BASE}/repos/{username}/{repo}/readme"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.raw"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None  # repo has no README
        print(f"  [error] {repo} -> HTTP {e.code}")
        return None
    except Exception as e:
        print(f"  [error] {repo} -> {e}")
        return None


def clean_readme(text: str, repo: str) -> str:
    """
    Light cleanup pass — strips the visual cruft that doesn't add
    retrieval value: badges, banner images, raw HTML, excess blank lines.
    Keeps all actual project description text intact.
    """
    # Remove HTML img/div/table tags but keep their text content where simple
    text = re.sub(r"<img[^>]*>", "", text)
    text = re.sub(r"</?div[^>]*>", "", text)
    text = re.sub(r"</?p[^>]*>", "", text)
    text = re.sub(r"<br\s*/?>", "\n", text)

    # Remove markdown badge/shield images: ![alt](url)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)

    # Remove markdown links that wrap only a badge/image (already stripped above,
    # this catches leftover [![...]](url) patterns)
    text = re.sub(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)", "", text)

    # Collapse 3+ blank lines into 1
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace on each line, drop now-empty badge lines
    lines = [line.rstrip() for line in text.split("\n")]
    lines = [line for line in lines if line.strip() != ""] or lines

    header = f"# {repo} — README\n\nSource: github.com/{GITHUB_USERNAME}/{repo}\n\n---\n\n"
    return header + "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull and clean GitHub READMEs")
    parser.add_argument("--user",  default=GITHUB_USERNAME, help="GitHub username")
    parser.add_argument("--repos", default="", help="Comma-separated repo names (default: all public repos)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.repos.strip():
        repos = [r.strip() for r in args.repos.split(",") if r.strip()]
        print(f"Fetching {len(repos)} specified repos for {args.user}...\n")
    else:
        print(f"Listing all public repos for {args.user}...")
        repos = list_repos(args.user)
        print(f"Found {len(repos)} repos.\n")

    saved, skipped = [], []

    for repo in repos:
        print(f"  {repo}...", end=" ")
        raw = fetch_readme_raw(args.user, repo)
        if raw is None:
            print("no README, skipped")
            skipped.append(repo)
            continue

        cleaned  = clean_readme(raw, repo)
        out_path = OUTPUT_DIR / f"{repo.lower().replace('-', '_')}_readme.md"
        out_path.write_text(cleaned, encoding="utf-8")
        print(f"saved -> {out_path}")
        saved.append(repo)

    print(f"\nDone. {len(saved)} saved, {len(skipped)} skipped (no README).")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
