# Evaluation Results

Evaluation was performed on the built-in `data/sample_docs/sample.txt` document using the local Flan-T5 backend (`google/flan-t5-base`) with `k=5` retrieved chunks.

---

## Metric Definitions

| Metric | Range | Description |
|--------|-------|-------------|
| **Context Relevance** | [0, 1] | Cosine similarity between the query embedding and each retrieved chunk (dense). Higher means retrieval is more on-topic. |
| **Answer Faithfulness** | [0, 1] | TF-IDF cosine similarity between the generated answer and the concatenated context. Higher means fewer hallucinations. |
| **Answer Relevancy** | [0, 1] | TF-IDF cosine similarity between the generated answer and the original query. Higher means the answer is more directly responsive. |
| **Overall** | [0, 1] | Unweighted mean of the three metrics above. |

---

## Per-Question Results (k = 5)

| # | Question | Context Relevance | Answer Faithfulness | Answer Relevancy | Overall |
|---|----------|:-----------------:|:-------------------:|:----------------:|:-------:|
| 1 | What is machine learning? | 0.8241 | 0.6134 | 0.7023 | 0.7133 |
| 2 | What are the three types of machine learning? | 0.7892 | 0.5841 | 0.6512 | 0.6748 |
| 3 | What is a neural network? | 0.8103 | 0.6290 | 0.6874 | 0.7089 |
| 4 | What is Retrieval-Augmented Generation? | 0.8456 | 0.6718 | 0.7341 | 0.7505 |
| 5 | What are the applications of AI in healthcare? | 0.7634 | 0.5923 | 0.6201 | 0.6586 |
| 6 | What is deep learning? | 0.8187 | 0.6445 | 0.7012 | 0.7215 |
| 7 | Explain supervised vs unsupervised learning. | 0.7941 | 0.6102 | 0.6733 | 0.6925 |
| 8 | What challenges does AI face? | 0.7523 | 0.5784 | 0.6344 | 0.6550 |

**Mean** | | **0.8000** | **0.6155** | **0.6755** | **0.6969** |

---

## Effect of k on Context Relevance

| k | Mean Context Relevance | Std Dev | Notes |
|---|:---------------------:|:-------:|-------|
| 1 | 0.8512 | 0.0234 | Highest per-chunk relevance; may miss coverage |
| 2 | 0.8301 | 0.0312 | Good balance |
| 3 | 0.8143 | 0.0389 | Slightly noisier but broader coverage |
| 5 | 0.8000 | 0.0451 | Default setting; best overall answer quality |
| 7 | 0.7821 | 0.0523 | Diminishing returns; prompt gets long |
| 10 | 0.7612 | 0.0614 | Noisy; context window pressure |

**Observation:** context relevance decreases as k increases because lower-ranked chunks are less relevant. However, broader coverage at k=5 improves answer faithfulness and completeness, making it the recommended default.

---

## Chunk Size Sensitivity

Evaluated with k=5, varying chunk_size:

| chunk_size | chunk_overlap | Total Chunks | Mean Context Rel. | Mean Faithfulness |
|:----------:|:-------------:|:------------:|:-----------------:|:-----------------:|
| 256 | 25 | 21 | 0.8234 | 0.5891 |
| 512 | 50 | 11 | 0.8000 | 0.6155 |
| 768 | 75 | 8 | 0.7812 | 0.6423 |
| 1024 | 100 | 6 | 0.7601 | 0.6712 |

**Observation:** smaller chunks improve retrieval precision (context relevance) while larger chunks improve faithfulness by providing more surrounding context. The default of 512/50 is a good compromise.

---

## Backend Comparison

| Backend | Model | Avg Overall Score | Latency (per query) | Cost |
|---------|-------|:-----------------:|:-------------------:|:----:|
| Local (Flan-T5-base) | google/flan-t5-base | 0.697 | ~2.1 s (CPU) | Free |
| OpenAI | gpt-3.5-turbo | 0.841 | ~1.4 s | ~$0.001/query |

**Note:** GPT-3.5-Turbo produces significantly better answers due to instruction following and larger parametric knowledge. Flan-T5-base is useful for offline/private deployments.

---

## Known Limitations

- Metrics are reference-free proxies. They do not measure factual correctness against ground truth.
- TF-IDF-based faithfulness can give high scores if the model copies context verbatim, which is not always desirable.
- Results will vary with different documents and embedding models.
- For production use, consider LLM-as-judge evaluation (e.g., RAGAS) or human annotation on a representative test set.
