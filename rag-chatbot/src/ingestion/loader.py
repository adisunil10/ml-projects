"""Document loading utilities for PDF and plain-text files."""

import os
from pathlib import Path
from typing import Any

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None  # type: ignore


class DocumentLoader:
    """Loads documents from PDF and plain-text sources."""

    def load_pdf(self, path: str | Path) -> list[dict[str, Any]]:
        """Load a PDF file and return a list of page documents.

        Args:
            path: Filesystem path to the PDF file.

        Returns:
            List of dicts with keys ``content`` and ``metadata``.
        """
        if PdfReader is None:
            raise ImportError("pypdf is required to load PDF files. Run: pip install pypdf")

        path = Path(path)
        reader = PdfReader(str(path))
        documents: list[dict[str, Any]] = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                documents.append(
                    {
                        "content": text,
                        "metadata": {
                            "source": str(path),
                            "filename": path.name,
                            "page": page_num + 1,
                            "total_pages": len(reader.pages),
                            "file_type": "pdf",
                        },
                    }
                )

        return documents

    def load_text(self, path: str | Path) -> list[dict[str, Any]]:
        """Load a plain-text file and return a single document.

        Args:
            path: Filesystem path to the text file.

        Returns:
            List with one dict containing ``content`` and ``metadata``.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read().strip()

        return [
            {
                "content": content,
                "metadata": {
                    "source": str(path),
                    "filename": path.name,
                    "page": 1,
                    "total_pages": 1,
                    "file_type": "txt",
                },
            }
        ]

    def load_directory(
        self,
        dir_path: str | Path,
        extensions: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Recursively load all supported documents from a directory.

        Args:
            dir_path: Root directory to scan.
            extensions: File extensions to include. Defaults to ``['.pdf', '.txt']``.

        Returns:
            Concatenated list of all loaded documents.
        """
        if extensions is None:
            extensions = [".pdf", ".txt"]

        dir_path = Path(dir_path)
        documents: list[dict[str, Any]] = []

        for ext in extensions:
            for file_path in sorted(dir_path.rglob(f"*{ext}")):
                if file_path.is_file():
                    if ext == ".pdf":
                        docs = self.load_pdf(file_path)
                    else:
                        docs = self.load_text(file_path)
                    documents.extend(docs)

        return documents
