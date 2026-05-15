"""Unit tests for TextChunker."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunker import TextChunker


SAMPLE_DOCUMENT = {
    "content": (
        "Artificial intelligence is the simulation of human intelligence processes by machines. "
        "These processes include learning, reasoning, and self-correction.\n\n"
        "Machine learning is a subset of artificial intelligence. "
        "It provides systems the ability to automatically learn and improve from experience "
        "without being explicitly programmed.\n\n"
        "Deep learning is part of a broader family of machine learning methods. "
        "It is based on artificial neural networks with representation learning. "
        "Learning can be supervised, semi-supervised or unsupervised.\n\n"
        "Natural language processing is a subfield of linguistics, computer science, "
        "and artificial intelligence concerned with the interactions between computers and human language."
    ),
    "metadata": {"source": "test.txt", "filename": "test.txt", "page": 1},
}


class TestTextChunker:
    def test_produces_chunks(self):
        chunker = TextChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_documents([SAMPLE_DOCUMENT])
        assert len(chunks) > 0

    def test_multiple_chunks_for_long_document(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_documents([SAMPLE_DOCUMENT])
        assert len(chunks) > 1

    def test_chunk_size_constraint(self):
        chunk_size = 150
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=10)
        chunks = chunker.chunk_documents([SAMPLE_DOCUMENT])
        # Most chunks should be under chunk_size (small overrun is tolerated at sentence boundaries)
        oversized = [c for c in chunks if len(c["content"]) > chunk_size * 1.5]
        assert len(oversized) == 0, f"Chunks exceed 1.5x the target size: {[len(c['content']) for c in oversized]}"

    def test_metadata_preserved(self):
        chunker = TextChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_documents([SAMPLE_DOCUMENT])
        for chunk in chunks:
            assert "metadata" in chunk
            assert chunk["metadata"]["source"] == "test.txt"
            assert chunk["metadata"]["filename"] == "test.txt"

    def test_chunk_index_in_metadata(self):
        chunker = TextChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_documents([SAMPLE_DOCUMENT])
        for chunk in chunks:
            assert "chunk_index" in chunk["metadata"]
            assert "chunk_count" in chunk["metadata"]
            assert chunk["metadata"]["chunk_index"] >= 0

    def test_empty_document_returns_no_chunks(self):
        chunker = TextChunker()
        chunks = chunker.chunk_documents([{"content": "", "metadata": {}}])
        assert chunks == []

    def test_single_short_document_is_one_chunk(self):
        chunker = TextChunker(chunk_size=1000, chunk_overlap=50)
        short_doc = {"content": "Hello world.", "metadata": {"source": "x.txt"}}
        chunks = chunker.chunk_documents([short_doc])
        assert len(chunks) == 1
        assert chunks[0]["content"] == "Hello world."

    def test_multiple_documents(self):
        chunker = TextChunker(chunk_size=200, chunk_overlap=20)
        docs = [SAMPLE_DOCUMENT, SAMPLE_DOCUMENT]
        chunks = chunker.chunk_documents(docs)
        single_count = len(chunker.chunk_documents([SAMPLE_DOCUMENT]))
        assert len(chunks) == single_count * 2

    def test_content_not_empty(self):
        chunker = TextChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_documents([SAMPLE_DOCUMENT])
        for chunk in chunks:
            assert chunk["content"].strip() != ""
