#!/usr/bin/env python3
"""CLI script to ingest documents into the RAG vector store.

Usage:
    python ingest.py --docs_dir ./data/sample_docs
    python ingest.py --docs_dir ./data/sample_docs --index_dir ./indexes/my_index
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ingestion.chunker import TextChunker
from src.ingestion.loader import DocumentLoader
from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import FAISSVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG vector store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--docs_dir",
        type=str,
        default="./data/sample_docs",
        help="Directory containing .pdf and/or .txt files to ingest.",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default="./indexes/default",
        help="Output directory where the FAISS index will be saved.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Target character length of each chunk.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=50,
        help="Character overlap between consecutive chunks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)

    if not docs_dir.exists():
        print(f"[error] Directory not found: {docs_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"=== RAG Chatbot — Document Ingestion ===")
    print(f"  Source : {docs_dir}")
    print(f"  Index  : {args.index_dir}")
    print(f"  Chunk size / overlap: {args.chunk_size} / {args.chunk_overlap}")
    print()

    loader = DocumentLoader()
    chunker = TextChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    embedder = Embedder()
    store = FAISSVectorStore()

    print("Loading documents ...")
    documents = loader.load_directory(docs_dir)
    if not documents:
        print("[warning] No documents found. Check the directory and file extensions.")
        sys.exit(0)
    print(f"  Loaded {len(documents)} document(s).")

    print("Chunking ...")
    chunks = chunker.chunk_documents(documents)
    print(f"  Created {len(chunks)} chunk(s).")

    print("Embedding ...")
    texts = [c["content"] for c in chunks]
    embeddings = embedder.embed_texts(texts)
    print(f"  Embedding shape: {embeddings.shape}")

    print("Indexing ...")
    store.add_documents(chunks, embeddings)

    print(f"Saving index to {args.index_dir} ...")
    store.save(args.index_dir)

    print()
    print(f"Done! Indexed {len(store)} vectors.")
    print(f"Run 'python app.py' to start the chatbot UI.")


if __name__ == "__main__":
    main()
