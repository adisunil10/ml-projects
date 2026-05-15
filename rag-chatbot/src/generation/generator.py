"""Answer generation via local Flan-T5 or the OpenAI Chat API."""

import os
from typing import Any, Literal


class Generator:
    """Generates answers from retrieved context chunks.

    Supports two backends:

    * ``"local"`` — uses ``google/flan-t5-base`` via HuggingFace Transformers.
      No API key required; runs entirely offline.
    * ``"openai"`` — uses ``gpt-3.5-turbo`` via the OpenAI API.
      Requires the ``OPENAI_API_KEY`` environment variable to be set.
    """

    LOCAL_MODEL = "google/flan-t5-base"
    MAX_NEW_TOKENS = 256

    def __init__(self, mode: Literal["local", "openai"] = "local") -> None:
        self.mode = mode
        self._local_pipeline = None
        self._openai_client = None

        if mode == "local":
            self._init_local()
        elif mode == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'local' or 'openai'.")

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _init_local(self) -> None:
        from transformers import pipeline

        self._local_pipeline = pipeline(
            "text2text-generation",
            model=self.LOCAL_MODEL,
            max_new_tokens=self.MAX_NEW_TOKENS,
        )

    def _init_openai(self) -> None:
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export it before using OpenAI mode."
            )
        self._openai_client = OpenAI(api_key=api_key)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(query: str, context_chunks: list[dict[str, Any]]) -> str:
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("metadata", {}).get("filename", "unknown")
            context_parts.append(f"[{i}] (source: {source})\n{chunk['content']}")
        context = "\n\n".join(context_parts)
        return (
            f"Answer based on context:\n\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, query: str, context_chunks: list[dict[str, Any]]) -> str:
        """Generate an answer for ``query`` grounded in ``context_chunks``.

        Args:
            query: The user's question.
            context_chunks: Retrieved document chunks (each with ``content`` key).

        Returns:
            A string answer.
        """
        if not context_chunks:
            return "I could not find relevant information to answer your question."

        prompt = self._build_prompt(query, context_chunks)

        if self.mode == "local":
            return self._generate_local(prompt)
        return self._generate_openai(prompt, query)

    def _generate_local(self, prompt: str) -> str:
        results = self._local_pipeline(prompt)
        return results[0]["generated_text"].strip()

    def _generate_openai(self, prompt: str, query: str) -> str:
        response = self._openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer questions accurately "
                        "using only the provided context. If the context does not "
                        "contain enough information, say so clearly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=self.MAX_NEW_TOKENS,
        )
        return response.choices[0].message.content.strip()
