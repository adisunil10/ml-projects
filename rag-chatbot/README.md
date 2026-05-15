# RAG Chatbot — End-to-End Retrieval-Augmented Generation

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)
![Code Style](https://img.shields.io/badge/code%20style-black-000000)
![Framework](https://img.shields.io/badge/UI-Gradio-orange?logo=gradio)
![Vector DB](https://img.shields.io/badge/vector%20store-FAISS-blueviolet)

A production-quality, fully-local **Retrieval-Augmented Generation** chatbot that lets you have natural conversations with your own documents — no external API required.

---

## Architecture

```
                        RAG Chatbot — System Architecture
  ┌──────────────────────────────────────────────────────────────────────┐
  │                         INGESTION PHASE                              │
  │                                                                      │
  │  ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌───────────┐  │
  │  │ Documents│───►│ DocumentLoader│───►│TextChunker│───►│  Embedder │  │
  │  │ PDF / TXT│    │ pypdf / open()│    │ recursive │    │MiniLM-L6  │  │
  │  └──────────┘    └──────────────┘    └──────────┘    └─────┬─────┘  │
  │                                                             │        │
  │                                                    float32 vectors   │
  │                                                             │        │
  │                                                    ┌────────▼──────┐ │
  │                                                    │ FAISSVectorStore│ │
  │                                                    │  IndexFlatIP  │ │
  │                                                    └───────────────┘ │
  └──────────────────────────────────────────────────────────────────────┘
                                    │
                              save / load
                                    │
  ┌─────────────────────────────────▼────────────────────────────────────┐
  │                          QUERY PHASE                                  │
  │                                                                       │
  │  User Query ──► Embedder ──► FAISS.search(k) ──► top-k chunks        │
  │                                                         │             │
  │                                               ┌─────────▼──────────┐ │
  │                                               │     Generator      │ │
  │                                               │  Flan-T5  / GPT   │ │
  │                                               └─────────┬──────────┘ │
  │                                                         │             │
  │  ┌───────────────┐         Answer + Sources ◄───────────┘             │
  │  │  RAGEvaluator │◄────────────────────────────────────────────────── │
  │  │  faithfulness │         scores dict                                │
  │  │  relevance    │                                                    │
  │  └───────────────┘                                                    │
  └───────────────────────────────────────────────────────────────────────┘
                                    │
                           ┌────────▼────────┐
                           │   Gradio UI     │
                           │  Chat + Upload  │
                           └─────────────────┘
```

---

## Features

- **Fully offline by default** — uses `google/flan-t5-base` for generation; no API key needed
- **OpenAI-compatible** — switch to `gpt-3.5-turbo` with a single flag for production-quality answers
- **Incremental ingestion** — upload and index new documents at runtime without rebuilding the index
- **Source attribution** — every answer is accompanied by the retrieved chunks and their similarity scores
- **Built-in evaluation** — three reference-free RAG metrics computed on every query
- **Persistent indexes** — save and reload FAISS indexes so you pay the embedding cost only once
- **Interactive UI** — clean Gradio chat interface with document upload, history, and source display
- **Jupyter notebooks** — exploratory data analysis, embedding visualisation, and evaluation sweeps
- **Full test suite** — unit tests for all components plus an integration test for the full pipeline

---

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Document Loading | `pypdf` | PDF text extraction |
| Text Chunking | Custom recursive splitter | Overlapping chunk segmentation |
| Embeddings | `sentence-transformers` | `all-MiniLM-L6-v2` (384-dim) |
| Vector Store | `faiss-cpu` | Fast k-NN search over dense vectors |
| Generation (local) | `transformers` | `google/flan-t5-base` text2text |
| Generation (cloud) | `openai` | `gpt-3.5-turbo` chat completions |
| Evaluation | `scikit-learn` | TF-IDF + cosine similarity metrics |
| UI | `gradio` | Interactive chat web interface |
| Acceleration | `torch` | PyTorch backend (CUDA / MPS / CPU) |

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourname/rag-chatbot.git
cd rag-chatbot

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install as a package
pip install -e .
```

For GPU acceleration, install the appropriate PyTorch build from https://pytorch.org/get-started/locally/ before running `pip install -r requirements.txt`.

---

## Quick Start

### Option A — Gradio UI (recommended)

```bash
# Load sample docs and launch the chat UI
python app.py
```

Open http://localhost:7860, click **Load Sample Documents**, then start chatting.

### Option B — CLI ingestion

```bash
# Ingest your own documents
python ingest.py --docs_dir ./data/sample_docs --index_dir ./indexes/my_index

# Available options
python ingest.py --help
```

### Option C — Python API

```python
from src.pipeline import RAGPipeline

# Build and ingest
pipeline = RAGPipeline(docs_dir="./data/sample_docs", mode="local", k=5)

# Query
result = pipeline.query("What is machine learning?")
print(result["answer"])
print(result["scores"])

# Each source chunk
for chunk in result["sources"]:
    print(f"[score={chunk['score']:.3f}] {chunk['content'][:100]}…")

# Save the index for later
pipeline.save_index("./indexes/my_index")

# Load it in a new session (no re-embedding needed)
pipeline2 = RAGPipeline(mode="local")
pipeline2.load_index("./indexes/my_index")
```

### OpenAI Mode

```bash
export OPENAI_API_KEY="sk-..."
python app.py
# Select "openai" from the Mode Selector in the UI
```

Or in Python:

```python
pipeline = RAGPipeline(docs_dir="./data/sample_docs", mode="openai")
```

---

## Project Structure

```
rag-chatbot/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── app.py                       # Gradio web UI
├── ingest.py                    # CLI ingestion script
├── src/
│   ├── pipeline.py              # RAGPipeline — end-to-end orchestration
│   ├── ingestion/
│   │   ├── loader.py            # PDF + TXT document loading
│   │   └── chunker.py           # Recursive text chunking with overlap
│   ├── embeddings/
│   │   └── embedder.py          # sentence-transformers embeddings
│   ├── retrieval/
│   │   └── vector_store.py      # FAISS vector store (add / search / save / load)
│   ├── generation/
│   │   └── generator.py         # Flan-T5 (local) + GPT-3.5 (OpenAI) backends
│   └── evaluation/
│       └── evaluator.py         # Reference-free RAG metrics
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_analysis.ipynb
│   └── 03_rag_evaluation.ipynb
├── tests/
│   ├── test_chunker.py
│   ├── test_embedder.py
│   └── test_pipeline.py
├── data/
│   └── sample_docs/
│       └── sample.txt           # AI/ML overview demo document
└── docs/
    ├── architecture.md
    └── evaluation_results.md
```

---

## How It Works

### 1. Ingestion
Documents are loaded from disk using `DocumentLoader`, which handles PDF (page-by-page via `pypdf`) and plain text. Each document is tagged with rich metadata (filename, page, file type).

### 2. Chunking
`TextChunker` recursively splits documents by paragraphs, then sentences, then character count — whichever boundary is cleanest. An overlap of `chunk_overlap` characters bridges adjacent chunks to preserve context across boundaries.

### 3. Embedding
`Embedder` encodes every chunk with `all-MiniLM-L6-v2`, producing a 384-dimensional, L2-normalised float32 vector per chunk. Device selection is automatic (CUDA > MPS > CPU).

### 4. Indexing
`FAISSVectorStore` stores chunk vectors in a `faiss.IndexFlatIP` index. Inner-product over normalised vectors is equivalent to cosine similarity, giving exact k-NN retrieval with no training overhead.

### 5. Retrieval
At query time, the question is embedded with the same model, and the top-k most similar chunks are retrieved from FAISS in milliseconds.

### 6. Generation
`Generator` constructs a grounded prompt that injects the retrieved chunks as numbered context blocks, then calls either Flan-T5-base (local) or GPT-3.5-Turbo (OpenAI) to produce a factual answer.

### 7. Evaluation
`RAGEvaluator` scores every query on three axes using only the pipeline's own outputs — no reference answers required.

---

## Evaluation Metrics

| Metric | Method | Interpretation |
|--------|--------|----------------|
| Context Relevance | Cosine sim (dense embeddings) | How on-topic is the retrieval? |
| Answer Faithfulness | TF-IDF cosine sim vs. context | Is the answer grounded in source material? |
| Answer Relevancy | TF-IDF cosine sim vs. query | Does the answer actually address the question? |

Typical scores on the built-in sample document with the local backend:

| Metric | Score |
|--------|-------|
| Context Relevance | 0.800 |
| Answer Faithfulness | 0.616 |
| Answer Relevancy | 0.676 |
| **Overall** | **0.697** |

See [docs/evaluation_results.md](docs/evaluation_results.md) for full result tables.

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific module
python -m pytest tests/test_chunker.py -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/ --cov=src --cov-report=term-missing
```

---

## Future Improvements

- **Hybrid retrieval:** combine dense FAISS search with BM25 sparse retrieval using Reciprocal Rank Fusion for better coverage on keyword queries
- **Cross-encoder reranking:** add a ColBERT or ms-marco reranker between retrieval and generation to improve precision
- **ANN indexing:** replace `IndexFlatIP` with `IndexHNSWFlat` or `IndexIVFFlat` for sub-linear search on millions of vectors
- **Streaming responses:** wire OpenAI's streaming API to Gradio's generator functions for real-time token display
- **Multi-modal ingestion:** extend `DocumentLoader` to handle images (via OCR) and HTML pages
- **Conversation memory:** add a sliding window of conversation turns to the generation prompt for multi-turn dialogue
- **LLM-as-judge evaluation:** integrate RAGAS or DeepEval for more rigorous, reference-based metric computation
- **Docker containerisation:** package the app with a `Dockerfile` for one-command deployment
- **Vector store swap:** add a `ChromaDB` or `Qdrant` backend adapter behind the `FAISSVectorStore` interface

---

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).

---

## Acknowledgements

- [sentence-transformers](https://www.sbert.net/) — `all-MiniLM-L6-v2` model
- [FAISS](https://github.com/facebookresearch/faiss) — Facebook AI Similarity Search
- [Gradio](https://gradio.app/) — rapid ML UI framework
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) — `flan-t5-base` model
