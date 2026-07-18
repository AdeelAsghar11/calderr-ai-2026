# Personal Knowledge Base

CalderR Week 3 · Project 3-I-A (Intermediate)

A RAG-powered Q&A system over my own documents: resume, portfolio, GitHub
project READMEs, CalderR internship materials, and hackathon writeups.
Ask a question, get a grounded answer with source citations.

---

## Architecture

```
                        INGESTION (run once)
┌──────────┐    ┌───────────┐    ┌───────────────┐    ┌───────────┐
│  docs/   │───▶│  loader   │───▶│  text splitter │───▶│ embedder  │
│ .pdf/.md │    │ (multi-   │    │  (512 char,    │    │(MiniLM-L6)│
│ .txt/    │    │  format)  │    │   50 overlap)  │    │           │
│ .docx    │    └───────────┘    └───────────────┘    └─────┬─────┘
└──────────┘                                                 │
                                                               ▼
                                                        ┌─────────────┐
                                                        │  ChromaDB   │
                                                        │ personal_kb │
                                                        └─────────────┘

                          QUERY (every question)
┌──────────┐    ┌───────────┐    ┌──────────────┐    ┌─────────────┐
│ question │───▶│  embed    │───▶│ vector search │───▶│  context    │
└──────────┘    │  (same    │    │ (+ optional   │    │  assembly   │
                 │   model)  │    │  source       │    │ (numbered,  │
                 └───────────┘    │  filter)      │    │  cited)     │
                                   └──────────────┘    └──────┬──────┘
                                                                │
                                                                ▼
                                                        ┌──────────────┐
                                                        │  Groq LLM    │
                                                        │ (streaming)  │
                                                        └──────┬───────┘
                                                                │
                                                                ▼
                                                        answer + sources
```

Two separate pipelines, same as any RAG system:
- **Ingestion** runs once — loads every file in `docs/`, chunks it, embeds it,
  stores it in ChromaDB.
- **Query** runs on every question — embeds the question with the same model,
  retrieves the top-k relevant chunks (optionally filtered to one document),
  and asks Groq to answer using only those chunks.

---

## Setup

```bash
pip install -r requirements.txt
```

Place your documents in `docs/` — PDF, Markdown, TXT, and DOCX are all
supported. Add your `GROQ_API_KEY` to a `.env` file in this folder.

```bash
# Ingest everything in docs/
python ingest.py run

# Check what's stored
python ingest.py stats
```

---

## Usage

### Ask a question

```bash
python kb.py ask "what was my BSL accuracy?"
python kb.py ask "what did I build for HACKDATA?" --show-context
```

### Filter to one document

```bash
python kb.py sources                                    # list all documents + slugs
python kb.py ask "what topics were covered?" --source calderr_week_2
```

### Interactive chat

```bash
python kb.py chat
# type /context to toggle showing retrieved sources
# type /exit to quit
```

### Generate the 15 Q&A examples deliverable

```bash
python generate_qa_examples.py
```
Saves a full transcript to `qa_examples.md` — 15 preset questions spanning
every document type in the knowledge base (resume, GitHub READMEs, all 4
CalderR weeks, both hackathon writeups), each with retrieved sources and
the generated answer.

---

## Skills demonstrated

- **Multi-format document loading** — a single loader handles PDF (pypdf),
  DOCX (python-docx), and plain text/Markdown, normalising all of them into
  the same `Document` object before chunking.
- **Embedding pipelines** — same embedding model (`all-MiniLM-L6-v2`) used
  at ingestion and query time, non-negotiable for retrieval to work at all.
- **ChromaDB** — own persistent collection (`personal_kb`), independent of
  the CalderR lab databases, with metadata-based source filtering.
- **Retrieval chains** — LangChain prompt template + Groq streaming, with
  a grounding constraint so answers only use retrieved context.

---

## Evaluation criteria (from project brief)

- ✅ **Retrieves relevant context** — hybrid of chunk-based search over
  20+ real personal documents, verifiable with `--show-context`.
- ✅ **Answers grounded in documents** — system prompt explicitly restricts
  the LLM to the provided context and instructs it to say so honestly when
  the context doesn't contain the answer.
- ✅ **Source citations shown** — every answer prints which document(s)
  grounded it; `--show-context` shows the exact retrieved chunks and
  distance scores.

---

## File structure

```
project-3-ia-personal-kb/
├── loader.py                # multi-format document loader
├── ingest.py                # chunk → embed → ChromaDB
├── kb.py                    # main CLI (ask, chat, sources)
├── generate_qa_examples.py  # produces the 15 Q&A deliverable
├── qa_examples.md           # generated output (15 Q&A transcript)
├── requirements.txt
├── README.md
├── docs/                    # your source documents (gitignore this)
└── kb_chroma_db/            # ChromaDB persistence (gitignore this)
```

Add to `.gitignore`:
```
docs/
kb_chroma_db/
```

(Or keep `docs/` tracked if you want the actual source documents in the
repo — your call, since this is personal content rather than sensitive data.)
