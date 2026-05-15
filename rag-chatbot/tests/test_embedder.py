"""Unit tests for the Embedder class."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    """Shared Embedder instance — loading the model once per test session."""
    return Embedder()


class TestEmbedder:
    def test_embed_texts_shape(self, embedder):
        texts = ["Hello world", "Machine learning is great", "I love Python"]
        embeddings = embedder.embed_texts(texts)
        assert embeddings.shape == (3, embedder.embedding_dim)

    def test_embed_texts_dtype(self, embedder):
        embeddings = embedder.embed_texts(["test"])
        assert embeddings.dtype == np.float32

    def test_embed_query_shape(self, embedder):
        vec = embedder.embed_query("What is AI?")
        assert vec.ndim == 1
        assert vec.shape[0] == embedder.embedding_dim

    def test_embedding_dim_matches_model(self, embedder):
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        assert embedder.embedding_dim == 384

    def test_normalized_embeddings(self, embedder):
        embeddings = embedder.embed_texts(["normalize me"])
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_similar_texts_high_cosine(self, embedder):
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "ML is part of AI and focuses on learning from data.",
        ]
        embs = embedder.embed_texts(texts)
        cosine_sim = float(np.dot(embs[0], embs[1]))
        assert cosine_sim > 0.7, f"Expected high similarity, got {cosine_sim:.4f}"

    def test_dissimilar_texts_lower_cosine(self, embedder):
        texts = [
            "The cat sat on the mat.",
            "Quantum chromodynamics describes the strong nuclear force.",
        ]
        embs = embedder.embed_texts(texts)
        cosine_sim = float(np.dot(embs[0], embs[1]))
        assert cosine_sim < 0.8, f"Expected lower similarity, got {cosine_sim:.4f}"

    def test_single_text_list(self, embedder):
        embs = embedder.embed_texts(["single"])
        assert embs.shape == (1, embedder.embedding_dim)

    def test_query_matches_embed_texts(self, embedder):
        query = "What is deep learning?"
        vec_from_query = embedder.embed_query(query)
        vec_from_texts = embedder.embed_texts([query])[0]
        np.testing.assert_allclose(vec_from_query, vec_from_texts, atol=1e-5)

    def test_batch_consistency(self, embedder):
        texts = ["Alpha", "Beta", "Gamma"]
        batch_embs = embedder.embed_texts(texts)
        for i, text in enumerate(texts):
            single = embedder.embed_query(text)
            np.testing.assert_allclose(batch_embs[i], single, atol=1e-5)
