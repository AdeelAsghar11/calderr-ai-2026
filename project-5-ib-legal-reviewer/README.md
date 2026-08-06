# Project 5-I-B: Multi-Agent Legal Document Reviewer

A portfolio-grade legal risk assessment and contract review system built with **Python**, **Pydantic**, **LangChain**, and **Groq (Llama 3.3 70B)**. The platform deploys four independent specialist review agents (Risk, Compliance, Liability, Obligations), a cross-examination debate facilitator, and a Chief Judge Agent to perform rigorous, adversarial legal document evaluations.

---

## 🏗️ System Architecture

The review workflow combines parallel independent analysis with structured adversarial peer debate and judicial synthesis:

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

## ⚖️ Debate & Judge Mechanism vs. Single-Pass Review

Single-pass LLM legal reviews frequently suffer from two critical flaws: **hallucinated high severity ratings** for routine administrative clauses, and **tunnel vision** where a single prompt misses cross-domain impacts.

The Multi-Agent Debate & Judge mechanism resolves these limitations through structured peer checks:

1. **Adversarial Calibration**: When one specialist overstates a clause's risk (e.g. rating a 24-hour notice period as severity 5), peer specialists challenge the rating during cross-examination with counter-arguments based on industry standard practices or statutory enforceability.
2. **Responsive Judicial Synthesis**: The Judge Agent does not treat initial ratings as static. When a finding receives a `dispute` challenge, the Judge marks it as `contested=True`, logs the peer dissent reasoning, and dynamically adjusts the final severity score.
3. **Auditability & Dissent Logs**: Unlike black-box reviews, every finding preserves a full transcript of peer consensus and dissent.

---

## 🧪 Sample Contracts Suite

The repository includes three synthetic contract documents in `sample_contracts/` containing realistic flaggable clauses:

| File Name | Document Type | Key Flaggable Clauses |
| :--- | :--- | :--- |
| `nda.txt` | Non-Disclosure Agreement | Broad confidential info definition, subpoena restriction, perpetual non-solicit, unilateral injunctive relief. |
| `consulting_agreement.txt` | Freelance/Consulting | Immediate termination without cause, IP assignment prior to payment, unlimited contractor liability, unannounced physical audits. |
| `saas_tos.txt` | SaaS Terms of Service | Unilateral modifications without notice, zero warranty/liability, 24-hour notice waiver period, offshore jurisdiction. |

---

## 🚀 Usage

### 1. Offline Smoke Test Suite
Run the deterministic offline smoke test suite (verifying independent review, debate disputes, and severity shifts without network calls):

```bash
uv run python project-5-ib-legal-reviewer/smoke_test.py
```

### 2. Typer CLI Reviewer
Run the Rich-formatted CLI on any contract file in offline stub mode or real LLM mode:

```bash
# Offline Stub Mode
uv run python project-5-ib-legal-reviewer/cli.py project-5-ib-legal-reviewer/sample_contracts/consulting_agreement.txt

# Real LLM Mode (requires GROQ_API_KEY)
uv run python project-5-ib-legal-reviewer/cli.py project-5-ib-legal-reviewer/sample_contracts/consulting_agreement.txt --real
```

### 3. Streamlit Interactive Dashboard
Launch the interactive web dashboard for inline contract visualization and debate inspection:

```bash
uv run streamlit run project-5-ib-legal-reviewer/dashboard.py --server.fileWatcherType=poll
```

---

## 🔑 Environment Setup

For online LLM execution (`--real` mode), set your Groq API key:


