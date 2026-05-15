# Architecture

## Overview

The RAG Chatbot is structured as a linear pipeline of five independent, composable modules. Each module has a single, well-defined responsibility. This design makes individual components easy to swap, test, and extend.

```
Raw Files
   │
   ▼
DocumentLoader          (src/ingestion/loader.py)
   │  reads PDF pages and plain-text files into a uniform dict schema
   ▼
TextChunker             (src/ingestion/chunker.py)
   │  recursively splits documents into overlapping fixed-size chunks
   ▼
Embedder                (src/embeddings/embedder.py)
   │  encodes chunk text into 384-dim float32 vectors (all-MiniLM-L6-v2)
   ▼
FAISSVectorStore        (src/retrieval/vector_store.py)
   │  stores vectors in a flat inner-product index; performs k-NN search
   ▼
Generator               (src/generation/generator.py)
   │  conditions a language model on retrieved context to produce an answer
   ▼
RAGEvaluator            (src/evaluation/evaluator.py)
      scores faithfulness, relevance, and grounding without gold labels
```

The `RAGPipeline` class (`src/pipeline.py`) wires these modules together and exposes two public methods: `ingest()` and `query()`.

---

## Module Details

### DocumentLoader

**File:** `src/ingestion/loader.py`

Responsible for converting files on disk into the internal document schema:

```python
{"content": str, "metadata": {"source": str, "filename": str, "page": int, ...}}
```

- PDF loading uses `pypdf.PdfReader`, which extracts text per page. Each page becomes a separate document so metadata (page number) is preserved.
- Text files are read with UTF-8 encoding (with `errors='replace'` for resilience) and returned as a single document.
- `load_directory()` uses `Path.rglob` to recursively find all matching files, making it easy to organise documents in subdirectories.

**Design decision:** keeping each PDF page as its own document (rather than joining the whole file) gives the chunker smaller, more coherent inputs and keeps page-number metadata accurate.

---

### TextChunker

**File:** `src/ingestion/chunker.py`

Implements a simple recursive splitting strategy:

1. If the text fits within `chunk_size`, return it as-is.
2. Otherwise, try splitting on double newlines (paragraph boundaries).
3. If that produces only one segment, fall back to sentence boundaries (`[.!?]` followed by whitespace).
4. If still indivisible, hard-split by character count.

Splits are greedily merged back into chunks up to `chunk_size`, then an overlap of `chunk_overlap` characters is carried forward into the next chunk. This ensures continuity across chunk boundaries.

`chunk_index` and `chunk_count` are injected into each chunk's metadata so the UI can display position context.

**Design decision:** avoiding dependencies on LangChain text splitters keeps the package lightweight and makes the algorithm fully transparent.

---

### Embedder

**File:** `src/embeddings/embedder.py`

Wraps `sentence-transformers` around the `all-MiniLM-L6-v2` checkpoint:

- **Dimensions:** 384
- **Speed:** ~14,000 sentences/second on CPU
- **Quality:** Competitive with larger models on semantic similarity benchmarks (MTEB)

Embeddings are L2-normalised by the SentenceTransformer library (`normalize_embeddings=True`), so cosine similarity equals inner product — which is what FAISS's `IndexFlatIP` computes.

Device detection order: CUDA → MPS (Apple Silicon) → CPU.

---

### FAISSVectorStore

**File:** `src/retrieval/vector_store.py`

Uses `faiss.IndexFlatIP` (exact inner-product search). For the scale of a portfolio project (thousands to tens-of-thousands of chunks), exact search is preferable to approximate search (HNSW/IVF) because it has no accuracy trade-off and requires no training.

Persistence splits into three files:
- `index.faiss` — the serialised FAISS index (binary)
- `chunks.pkl` — the list of chunk dicts (Python pickle)
- `meta.json` — human-readable metadata (dimension, vector count)

**Design decision:** separating chunk storage from the FAISS index allows the index to be rebuilt from different embedding models without losing the original text.

---

### Generator

**File:** `src/generation/generator.py`

Supports two backends selectable at construction time:

| Mode | Model | Requirements |
|------|-------|--------------|
| `local` | `google/flan-t5-base` | None (downloads once via HuggingFace Hub) |
| `openai` | `gpt-3.5-turbo` | `OPENAI_API_KEY` environment variable |

The prompt template is deliberately minimal:

```
Answer based on context:

[1] (source: doc.pdf)
<chunk text>

[2] (source: doc.txt)
<chunk text>

Question: <user query>

Answer:
```

Source attribution is included in the prompt so the model can reference specific documents. Temperature is set to 0.2 for OpenAI to reduce hallucination.

---

### RAGEvaluator

**File:** `src/evaluation/evaluator.py`

Three reference-free metrics, all bounded in [0, 1]:

| Metric | Method | Measures |
|--------|--------|----------|
| `context_relevance` | Cosine sim (dense or TF-IDF) between query and chunks | Retrieval quality |
| `answer_faithfulness` | TF-IDF cosine sim between answer and context | Grounding |
| `answer_relevancy` | TF-IDF cosine sim between answer and query | Response quality |

When an `Embedder` is provided, `context_relevance` uses dense vectors (more accurate). The other two metrics always use TF-IDF because the answer and context are often lexically overlapping, making sparse similarity a better signal there.

**Limitations:** these are proxy metrics. For production evaluation, consider RAGAS with LLM-as-judge or human annotation.

---

### RAGPipeline

**File:** `src/pipeline.py`

The top-level orchestrator. Key design choices:

- **Lazy generator init:** the `Generator` loads the model in `__init__`, so the first import pays the cost once. Subsequent queries are fast.
- **Incremental ingestion:** `add_documents_from_files()` appends to an existing FAISS index without rebuilding it, enabling live document uploads in the UI.
- **Index persistence:** `save_index()` / `load_index()` allow pre-built indexes to be shipped with the project, so users don't have to re-embed on every startup.

---

## Data Flow Diagram

```
User Question ──────────────────────────────────┐
                                                 │
                                                 ▼
                                          Embedder.embed_query()
                                                 │
                                                 ▼
                                     FAISSVectorStore.search(k)
                                                 │
                        ┌────────────────────────┘
                        │  top-k chunks
                        ▼
                   Generator.generate()
                        │
                        ▼
                      Answer ──► RAGEvaluator.evaluate() ──► Scores
                        │
                        ▼
                  {"answer", "sources", "scores"}
```

---

## Extension Points

- **Swap the embedding model:** change `Embedder.MODEL_NAME`. The FAISS index dimension is stored in metadata, so a mismatch will raise an informative error.
- **Add reranking:** insert a cross-encoder between `vector_store.search()` and `generator.generate()` in `pipeline.py`.
- **Use ANN for scale:** replace `faiss.IndexFlatIP` with `faiss.IndexHNSWFlat` for sub-linear search on millions of vectors.
- **Add streaming:** the OpenAI backend supports `stream=True`; wire it to a Gradio `gr.ChatInterface` generator function.
- **Hybrid search:** combine FAISS dense search with BM25 sparse search using Reciprocal Rank Fusion (RRF).
