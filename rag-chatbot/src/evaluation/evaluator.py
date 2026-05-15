"""RAG evaluation metrics using cosine similarity and TF-IDF."""

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGEvaluator:
    """Lightweight, reference-free RAG evaluation using sklearn.

    Metrics implemented:

    * **context_relevance** — average cosine similarity between the query
      embedding and each retrieved chunk embedding (measures retrieval quality).
    * **answer_faithfulness** — TF-IDF cosine similarity between the generated
      answer and the concatenated context (measures grounding).
    * **answer_relevancy** — TF-IDF cosine similarity between the generated
      answer and the original query (measures response quality).
    """

    def __init__(self, embedder=None) -> None:
        """
        Args:
            embedder: An ``Embedder`` instance.  If ``None``, context_relevance
                      will fall back to TF-IDF-based similarity.
        """
        self.embedder = embedder
        self._tfidf = TfidfVectorizer(stop_words="english")

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def context_relevance(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> float:
        """Measure how relevant the retrieved chunks are to the query.

        Uses dense embeddings when an embedder is available, otherwise TF-IDF.

        Returns:
            Mean cosine similarity in ``[0, 1]``.
        """
        if not retrieved_chunks:
            return 0.0

        chunk_texts = [c["content"] for c in retrieved_chunks]

        if self.embedder is not None:
            query_emb = self.embedder.embed_query(query).reshape(1, -1)
            chunk_embs = self.embedder.embed_texts(chunk_texts)
            sims = cosine_similarity(query_emb, chunk_embs)[0]
        else:
            corpus = [query] + chunk_texts
            tfidf_matrix = self._tfidf.fit_transform(corpus)
            sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

        return float(np.mean(sims))

    def answer_faithfulness(
        self,
        answer: str,
        context_chunks: list[dict[str, Any]],
    ) -> float:
        """Measure how well the answer is grounded in the context.

        Returns:
            Cosine similarity between answer and concatenated context TF-IDF
            vectors, in ``[0, 1]``.
        """
        if not context_chunks or not answer.strip():
            return 0.0

        context_text = " ".join(c["content"] for c in context_chunks)
        corpus = [answer, context_text]

        try:
            tfidf_matrix = self._tfidf.fit_transform(corpus)
            sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0][0]
        except ValueError:
            sim = 0.0

        return float(sim)

    def answer_relevancy(self, query: str, answer: str) -> float:
        """Measure how relevant the answer is to the original question.

        Returns:
            Cosine similarity between answer and query TF-IDF vectors, in ``[0, 1]``.
        """
        if not answer.strip() or not query.strip():
            return 0.0

        try:
            tfidf_matrix = self._tfidf.fit_transform([query, answer])
            sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0][0]
        except ValueError:
            sim = 0.0

        return float(sim)

    # ------------------------------------------------------------------
    # Composite evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
        answer: str,
    ) -> dict[str, float]:
        """Run all metrics and return a combined score dictionary.

        Args:
            query: The user question.
            retrieved_chunks: Chunks returned by retrieval.
            answer: The generated answer.

        Returns:
            Dict with keys ``context_relevance``, ``answer_faithfulness``,
            ``answer_relevancy``, and ``overall`` (unweighted mean).
        """
        cr = self.context_relevance(query, retrieved_chunks)
        af = self.answer_faithfulness(answer, retrieved_chunks)
        ar = self.answer_relevancy(query, answer)
        overall = (cr + af + ar) / 3.0

        return {
            "context_relevance": round(cr, 4),
            "answer_faithfulness": round(af, 4),
            "answer_relevancy": round(ar, 4),
            "overall": round(overall, 4),
        }
