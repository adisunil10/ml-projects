"""End-to-end RAG pipeline wiring ingestion, retrieval, and generation."""

import sys
from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.embedder import Embedder
from src.evaluation.evaluator import RAGEvaluator
from src.generation.generator import Generator
from src.ingestion.chunker import TextChunker
from src.ingestion.loader import DocumentLoader
from src.retrieval.vector_store import FAISSVectorStore


class RAGPipeline:
    """Full retrieval-augmented generation pipeline.

    Usage::

        pipeline = RAGPipeline(docs_dir="./data/sample_docs", mode="local")
        result = pipeline.query("What is machine learning?")
        print(result["answer"])
        print(result["sources"])
    """

    def __init__(
        self,
        docs_dir: str | Path | None = None,
        mode: Literal["local", "openai"] = "local",
        k: int = 5,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        """
        Args:
            docs_dir: Directory to ingest on construction. Pass ``None`` to
                      defer ingestion (e.g. when loading a saved index).
            mode: Generation backend — ``"local"`` or ``"openai"``.
            k: Number of chunks to retrieve per query.
            chunk_size: Target character length of each chunk.
            chunk_overlap: Overlap between consecutive chunks.
        """
        self.k = k
        self.mode = mode

        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = Embedder()
        self.vector_store = FAISSVectorStore()
        self.generator = Generator(mode=mode)
        self.evaluator = RAGEvaluator(embedder=self.embedder)

        if docs_dir is not None:
            self.ingest(docs_dir)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, docs_dir: str | Path) -> int:
        """Load, chunk, embed, and index all documents in ``docs_dir``.

        Args:
            docs_dir: Directory containing ``.pdf`` and/or ``.txt`` files.

        Returns:
            Number of chunks indexed.
        """
        docs_dir = Path(docs_dir)
        print(f"[ingest] Loading documents from {docs_dir} ...")
        documents = self.loader.load_directory(docs_dir)
        print(f"[ingest] Loaded {len(documents)} document(s).")

        print("[ingest] Chunking ...")
        chunks = self.chunker.chunk_documents(documents)
        print(f"[ingest] Created {len(chunks)} chunk(s).")

        print("[ingest] Embedding ...")
        texts = [c["content"] for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        print("[ingest] Indexing ...")
        self.vector_store.add_documents(chunks, embeddings)
        print(f"[ingest] Done. {len(self.vector_store)} vectors in store.")

        return len(chunks)

    def add_documents_from_files(self, file_paths: list[str | Path]) -> int:
        """Incrementally add documents from a list of file paths."""
        documents: list[dict[str, Any]] = []
        for fp in file_paths:
            fp = Path(fp)
            if fp.suffix.lower() == ".pdf":
                documents.extend(self.loader.load_pdf(fp))
            elif fp.suffix.lower() == ".txt":
                documents.extend(self.loader.load_text(fp))

        if not documents:
            return 0

        chunks = self.chunker.chunk_documents(documents)
        texts = [c["content"] for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        self.vector_store.add_documents(chunks, embeddings)
        return len(chunks)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, question: str) -> dict[str, Any]:
        """Answer a question using the indexed documents.

        Args:
            question: The user's natural-language question.

        Returns:
            Dict with keys:
            * ``answer`` — generated string answer.
            * ``sources`` — list of retrieved chunk dicts (with ``score``).
            * ``scores`` — evaluation metrics dict.
        """
        if len(self.vector_store) == 0:
            return {
                "answer": "No documents have been indexed yet. Please ingest documents first.",
                "sources": [],
                "scores": {},
            }

        query_embedding = self.embedder.embed_query(question)
        retrieved_chunks = self.vector_store.search(query_embedding, k=self.k)

        answer = self.generator.generate(question, retrieved_chunks)
        scores = self.evaluator.evaluate(question, retrieved_chunks, answer)

        return {
            "answer": answer,
            "sources": retrieved_chunks,
            "scores": scores,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self, path: str | Path) -> None:
        """Save the FAISS index and chunk metadata to ``path``."""
        self.vector_store.save(path)
        print(f"[pipeline] Index saved to {path}")

    def load_index(self, path: str | Path) -> None:
        """Load a previously saved index from ``path``."""
        self.vector_store = FAISSVectorStore.load(path)
        print(f"[pipeline] Index loaded from {path} ({len(self.vector_store)} vectors)")
