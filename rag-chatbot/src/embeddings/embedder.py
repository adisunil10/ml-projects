"""Sentence-transformer embeddings with automatic device selection."""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class Embedder:
    """Wraps a HuggingFace sentence-transformers model for text embedding.

    Uses ``all-MiniLM-L6-v2`` by default, which produces 384-dimensional
    embeddings at high speed with competitive quality.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or self.MODEL_NAME
        self.device = self._detect_device()
        self.model = SentenceTransformer(self.model_name, device=self.device)

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings.

        Args:
            texts: Input strings to encode.

        Returns:
            Float32 numpy array of shape ``(len(texts), embedding_dim)``.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Args:
            query: The question or search string.

        Returns:
            Float32 numpy array of shape ``(embedding_dim,)``.
        """
        return self.embed_texts([query])[0]

    @property
    def embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
