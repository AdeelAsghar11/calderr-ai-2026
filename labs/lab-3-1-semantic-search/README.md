# Lab 3.1 — Semantic Search CLI

CalderR Agentic AI Engineering Internship · Week 3 · Monday

---

## What this lab covers

**Embeddings** turn text into vectors — lists of numbers where the *direction*
encodes meaning.  Two sentences about space exploration point in roughly the
same direction in this high-dimensional space, even if they share no words.

**Cosine similarity** measures the angle between two vectors, not their length.
Sentences that mean similar things have a small angle (high cosine, close to 1).
Unrelated sentences point in different directions (cosine near 0).

This lab builds a CLI that:
1. Embeds 100 diverse Wikipedia sentences using two models
2. Takes a user query, embeds it the same way
3. Returns the most semantically similar sentences by cosine score
4. Compares results and agreement between `all-MiniLM-L6-v2` and `bge-small-en`
5. Visualises the embedding space with PCA

---

## Core concepts

### What is an embedding?
An embedding model maps text → a fixed-length vector (here: 384 dimensions).
The model has learned during training that similar meanings → similar directions.

```
"black holes in space"   → [0.23, -0.41, 0.17, ... 384 values]
"astronomical objects"   → [0.21, -0.38, 0.19, ... 384 values]  ← nearly parallel
"history of the Romans"  → [-0.12, 0.55, -0.30, ...]            ← very different angle
```

### Why cosine similarity and not Euclidean distance?
In high-dimensional spaces, Euclidean distance is dominated by vector *magnitude*
(long sentences produce large vectors).  Cosine measures only the *angle*, so it
compares meaning regardless of sentence length.

```
cos(θ) = (a · b) / (||a|| × ||b||)
```

With `normalize_embeddings=True`, every vector has magnitude 1, so `cos(θ) = a · b`
(just a dot product — fast, vectorisable, identical result).

### MiniLM vs BGE
| Model | Full name | Strength |
|---|---|---|
| `minilm` | `all-MiniLM-L6-v2` | General-purpose, fast, good baseline |
| `bge` | `BAAI/bge-small-en-v1.5` | Asymmetric retrieval; needs a query prefix |

BGE uses asymmetric retrieval — query and corpus are encoded differently.
The query prefix tells the model "this is a search query, not a passage."
This is implemented automatically in `embedder.py`.

---

## Setup

```bash
pip install -r requirements.txt
```

First run downloads the model weights (~90 MB for MiniLM, ~130 MB for BGE)
and caches embeddings to `.cache/` — subsequent runs are instant.

---

## Commands

### `search` — single model search
```bash
python main.py search "history of ancient civilisations"
python main.py search "space exploration" --model bge --top-k 8
python main.py search "deep learning and neural networks" -m minilm -k 10
```

### `compare` — side-by-side model comparison
```bash
python main.py compare "treatment of infectious diseases"
python main.py compare "famous painters and artwork" --top-k 3
```

### `visualize` — PCA scatter plot
```bash
python main.py visualize                   # uses MiniLM, saves embeddings_pca.png
python main.py visualize --model bge -o bge_pca.png
```

### `info` — corpus and cache status
```bash
python main.py info
```

---

## Example queries to try

| Query | Expected top results |
|---|---|
| `"space exploration and rockets"` | Apollo Moon landing, Voyager 1, Mars |
| `"treatment of bacterial infections"` | Penicillin, vaccines, COVID |
| `"history of computing and the internet"` | ENIAC, ARPANET, Python |
| `"economic recession and financial crisis"` | Great Depression, GDP, supply/demand |
| `"visual artists and painting techniques"` | Van Gogh, Picasso, Michelangelo |
| `"genetics and DNA research"` | CRISPR, human genome, double helix |

---

## File structure

```
lab-3-1-semantic-search/
├── main.py          # Typer CLI (search, compare, visualize, info)
├── embedder.py      # SemanticEmbedder class with cosine search
├── sentences.py     # 100 Wikipedia sentences across 20 categories
├── requirements.txt
├── README.md
└── .cache/          # Numpy embedding cache (gitignored)
    ├── embeddings_minilm.npy
    └── embeddings_bge.npy
```

---

## What to observe

- **High agreement** (4–5/5 shared) on unambiguous queries like `"space astronomy"`
- **Low agreement** (1–2/5 shared) on ambiguous queries like `"Greek culture"` (History vs Arts)
- BGE sometimes surfaces more relevant results on technical queries due to asymmetric design
- PCA plot: science sentences cluster together, arts cluster separately; history spreads wide
