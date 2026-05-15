"""Recursive text chunking with configurable size and overlap."""

import re
from typing import Any


class TextChunker:
    """Splits documents into overlapping chunks using recursive paragraph/sentence splitting."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        """
        Args:
            chunk_size: Target maximum character length of each chunk.
            chunk_overlap: Number of characters to overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_text(self, text: str) -> list[str]:
        """Recursively split text by paragraphs then sentences."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Try splitting by double newline (paragraphs)
        paragraphs = re.split(r"\n\n+", text)
        if len(paragraphs) > 1:
            return self._merge_splits(paragraphs)

        # Fall back to sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 1:
            return self._merge_splits(sentences)

        # Last resort: hard split by character count
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.chunk_overlap if end - self.chunk_overlap > start else end
        return chunks

    def _merge_splits(self, splits: list[str]) -> list[str]:
        """Greedily merge small splits into chunks up to chunk_size, with overlap."""
        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for part in splits:
            part = part.strip()
            if not part:
                continue
            part_len = len(part)

            if current_len + part_len + 1 > self.chunk_size and current_parts:
                chunk_text = " ".join(current_parts).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                # Keep trailing parts for overlap
                overlap_text = chunk_text[-self.chunk_overlap :] if self.chunk_overlap else ""
                current_parts = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)

            current_parts.append(part)
            current_len += part_len + 1

        if current_parts:
            remaining = " ".join(current_parts).strip()
            if remaining:
                chunks.append(remaining)

        return chunks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Chunk a list of documents into smaller pieces, preserving metadata.

        Args:
            documents: List of dicts with ``content`` and ``metadata`` keys.

        Returns:
            List of chunk dicts with ``content``, ``metadata``, and ``chunk_index``.
        """
        all_chunks: list[dict[str, Any]] = []

        for doc in documents:
            content: str = doc.get("content", "")
            metadata: dict[str, Any] = doc.get("metadata", {})

            raw_chunks = self._split_text(content)

            for idx, chunk_text in enumerate(raw_chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue
                all_chunks.append(
                    {
                        "content": chunk_text,
                        "metadata": {
                            **metadata,
                            "chunk_index": idx,
                            "chunk_count": len(raw_chunks),
                        },
                    }
                )

        return all_chunks
