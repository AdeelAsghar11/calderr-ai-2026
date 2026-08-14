# Project 5-I-B: Multi-Agent Legal Document Reviewer

A portfolio-grade legal risk assessment and contract review system built with **Python 3.11+**, **Pydantic**, **LangChain**, **Groq (`llama-3.3-70b-versatile`)**, **Streamlit**, and **Typer**.

The platform deploys four independent specialist review agents (Risk, Compliance, Liability, Obligations), a cross-examination debate facilitator, and a Chief Judge Agent to perform rigorous, adversarial legal document evaluations.

---

## 🏗️ System Architecture

```
                               ┌─────────────────────────┐
                               │    Contract Document    │
                               └────────────┬────────────┘
                                            │
                ┌──────────────────────────┴──────────────────────────┐
                ▼                          ▼                          ▼                          ▼
      ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
      │    Risk Agent     │      │ Compliance Agent  │      │  Liability Agent  │      │ Obligations Agent │
      └─────────┬─────────┘      └─────────┬─────────┘      └─────────┬─────────┘      └─────────┬─────────┘
                │                          │                          │                          │
                └──────────────────────────┼──────────────────────────┘                          │
                                           ▼                                                     │
                              ┌─────────────────────────┐                                        │
                              │   Independent Findings  │◄───────────────────────────────────────┘
                              └────────────┬────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │   Debate Facilitator    │
                              │   (Cross-Examination) │
                              └────────────┬────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │  Peer Challenges Log   │
                              │  (Agree vs Dispute)     │
                              └────────────┬────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │       Judge Agent       │
                              │  (Severity & Synthesis) │
                              └────────────┬────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │   Final ReviewReport    │
                              └─────────────────────────┘
```

### Roles & Responsibilities

- **Risk Agent**: Identifies unfavorable terms, unilateral rights, and missing structural protections.
- **Compliance Agent**: Checks for regulatory red flags, missing statutory carve-outs, and legal compliance risks.
- **Liability Agent**: Maps liability exposure, indemnity imbalances, and uncapped financial risks.
- **Obligations Agent**: Extracts strict performance deadlines, operational burdens, and audit triggers.
- **Debate Facilitator**: A deterministic workflow dispatcher that routes every finding raised by a specialist to the three other peer specialists for structured cross-examination (`agree` or `dispute` with legal rationale).
- **Judge Agent**: Evaluates initial findings alongside every peer challenge, assigns final severity scores (1-5), flags contested findings, logs dissent notes, and compiles the final executive report.

---

## 🧪 Sample Contracts Suite

The repository includes synthetic contract documents in `sample_contracts/` containing realistic flaggable clauses:

| File Name | Document Type | Key Flaggable Clauses |
| :--- | :--- | :--- |
| `nda.txt` | Non-Disclosure Agreement | Broad confidential info definition, subpoena restriction, perpetual non-solicit, unilateral injunctive relief. |
| `consulting_agreement.txt` | Freelance/Consulting | Immediate termination without cause, IP assignment prior to payment, unlimited contractor liability, unannounced physical audits. |
| `saas_tos.txt` | SaaS Terms of Service | Unilateral modifications without notice, zero warranty/liability, 24-hour notice waiver period, offshore jurisdiction. |

---

## 🚀 Quick Start & Usage

### Prerequisites
- Python 3.11+
- Groq API Key set in `.env` (`GROQ_API_KEY`)

### 1. Offline Smoke Test Suite
Run deterministic offline tests:
```bash
uv run python projects/project-5-ib-legal-reviewer/smoke_test.py
```

### 2. Typer CLI Reviewer
Run the Rich-formatted CLI on any contract file:
```bash
# Offline Stub Mode
uv run python projects/project-5-ib-legal-reviewer/cli.py projects/project-5-ib-legal-reviewer/sample_contracts/consulting_agreement.txt

# Real LLM Mode (requires GROQ_API_KEY)
uv run python projects/project-5-ib-legal-reviewer/cli.py projects/project-5-ib-legal-reviewer/sample_contracts/consulting_agreement.txt --real
```

### 3. Streamlit Interactive Dashboard
Launch the web dashboard for contract inspection and debate visualization:
```bash
uv run streamlit run projects/project-5-ib-legal-reviewer/dashboard.py
```

---

## 📂 Directory Structure

```text
projects/project-5-ib-legal-reviewer/
├── README.md               # Architecture & documentation
├── cli.py                  # Typer CLI application
├── dashboard.py            # Streamlit interactive interface
├── legal_reviewer.py       # Core multi-agent review & debate engine
├── models.py               # Pydantic data schemas
├── sample_contracts/       # Test contracts (NDA, Consulting, SaaS TOS)
└── smoke_test.py           # Offline verification suite
```

---

## 💻 Tech Stack

- **LLM Engine:** Groq API (`llama-3.3-70b-versatile`)
- **Frontend UI:** Streamlit & Typer CLI
- **Framework:** LangChain & Pydantic v2
