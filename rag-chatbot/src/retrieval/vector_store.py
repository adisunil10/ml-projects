"""FAISS-backed vector store for dense retrieval."""

import json
import os
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class FAISSVectorStore:
    """Stores document chunks alongside a FAISS index for fast similarity search.

    The index uses inner-product (IP) search over L2-normalised embeddings,
    which is equivalent to cosine similarity.
    """

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.chunks: list[dict[str, Any]] = []
        self._dim: int | None = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(
        self,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """Add chunks and their embeddings to the store.

        Args:
            chunks: List of chunk dicts (must have ``content`` and ``metadata``).
            embeddings: Float32 array of shape ``(len(chunks), dim)``.
        """
        embeddings = embeddings.astype(np.float32)
        dim = embeddings.shape[1]

        if self.index is None:
            self._dim = dim
            self.index = faiss.IndexFlatIP(dim)

        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top-k most similar chunks for a query embedding.

        Args:
            query_embedding: Float32 array of shape ``(dim,)``.
            k: Number of results to return.

        Returns:
            List of dicts, each containing the original chunk keys plus a
            ``score`` key (cosine similarity, higher is better).
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)
        vec = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)

        scores, indices = self.index.search(vec, k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the FAISS index and chunk metadata to disk.

        Args:
            path: Directory path where files will be written.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))

        with open(path / "chunks.pkl", "wb") as fh:
            pickle.dump(self.chunks, fh)

        with open(path / "meta.json", "w") as fh:
            json.dump({"dim": self._dim, "total": self.index.ntotal}, fh)

    @classmethod
    def load(cls, path: str | Path) -> "FAISSVectorStore":
        """Load a previously saved vector store from disk.

        Args:
            path: Directory path containing the saved files.

        Returns:
            A fully initialised ``FAISSVectorStore`` instance.
        """
        path = Path(path)
        store = cls()

        store.index = faiss.read_index(str(path / "index.faiss"))

        with open(path / "chunks.pkl", "rb") as fh:
            store.chunks = pickle.load(fh)

        with open(path / "meta.json") as fh:
            meta = json.load(fh)
        store._dim = meta["dim"]

        return store

    def __len__(self) -> int:
        return self.index.ntotal if self.index else 0
