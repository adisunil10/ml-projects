#!/usr/bin/env python3
"""Gradio chat UI for the RAG chatbot.

Run:
    python app.py
"""

import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from src.pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# Global state — one pipeline instance shared across sessions
# ---------------------------------------------------------------------------
_pipeline: RAGPipeline | None = None


def get_pipeline(mode: str) -> RAGPipeline:
    global _pipeline
    if _pipeline is None or _pipeline.mode != mode:
        _pipeline = RAGPipeline(mode=mode)
    return _pipeline


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def upload_documents(files: list[Any], mode: str) -> str:
    """Ingest uploaded files into the vector store."""
    if not files:
        return "No files uploaded."

    pipeline = get_pipeline(mode)
    file_paths = [Path(f.name) for f in files]
    n = pipeline.add_documents_from_files(file_paths)
    names = ", ".join(p.name for p in file_paths)
    return f"Ingested {n} chunk(s) from: {names}\nTotal indexed: {len(pipeline.vector_store)} vectors."


def load_sample_docs(mode: str) -> str:
    """Load the built-in sample documents."""
    pipeline = get_pipeline(mode)
    sample_dir = Path(__file__).parent / "data" / "sample_docs"
    if not sample_dir.exists():
        return "Sample docs directory not found."
    n = pipeline.ingest(sample_dir)
    return f"Loaded sample documents: {n} chunk(s) indexed."


def chat(
    message: str,
    history: list[tuple[str, str]],
    mode: str,
) -> tuple[list[tuple[str, str]], str]:
    """Process a user message and return updated history + sources."""
    if not message.strip():
        return history, ""

    pipeline = get_pipeline(mode)
    result = pipeline.query(message)

    answer = result["answer"]
    sources = result["sources"]
    scores = result["scores"]

    # Format sources panel
    if sources:
        source_lines = ["**Retrieved Sources:**\n"]
        for i, chunk in enumerate(sources[:3], 1):
            meta = chunk.get("metadata", {})
            fname = meta.get("filename", "unknown")
            score = chunk.get("score", 0.0)
            preview = chunk["content"][:200].replace("\n", " ") + "..."
            source_lines.append(
                f"**[{i}] {fname}** (score: {score:.3f})\n> {preview}\n"
            )
        if scores:
            source_lines.append(
                f"\n**Evaluation Scores:**  "
                f"Context Relevance: `{scores.get('context_relevance', 0):.3f}` | "
                f"Faithfulness: `{scores.get('answer_faithfulness', 0):.3f}` | "
                f"Answer Relevancy: `{scores.get('answer_relevancy', 0):.3f}`"
            )
        sources_text = "\n".join(source_lines)
    else:
        sources_text = "No sources retrieved."

    history = list(history) + [(message, answer)]
    return history, sources_text


def clear_chat() -> tuple[list, str, str]:
    return [], "", ""


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="RAG Chatbot",
        theme=gr.themes.Soft(),
        css=".source-box { font-size: 0.85rem; }",
    ) as demo:
        gr.Markdown(
            """
            # RAG Chatbot
            **Retrieval-Augmented Generation** — ask questions about your documents.
            Upload PDFs or text files, or load the built-in sample documents to get started.
            """
        )

        with gr.Row():
            # Left column: configuration + upload
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")
                mode_selector = gr.Radio(
                    choices=["local", "openai"],
                    value="local",
                    label="Generation Mode",
                    info="'local' uses Flan-T5 (no API key needed). 'openai' uses GPT-3.5-Turbo.",
                )

                gr.Markdown("### Documents")
                sample_btn = gr.Button("Load Sample Documents", variant="secondary")
                file_upload = gr.File(
                    label="Upload your own documents (.pdf, .txt)",
                    file_types=[".pdf", ".txt"],
                    file_count="multiple",
                )
                upload_btn = gr.Button("Ingest Uploaded Files", variant="primary")
                ingest_status = gr.Textbox(
                    label="Ingestion Status",
                    interactive=False,
                    lines=3,
                )

                gr.Markdown("### Sources")
                sources_display = gr.Markdown(
                    value="Sources will appear here after a query.",
                    elem_classes=["source-box"],
                )

            # Right column: chat
            with gr.Column(scale=2):
                gr.Markdown("### Chat")
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=450,
                    show_copy_button=True,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask a question about your documents...",
                        label="Your question",
                        scale=4,
                        lines=1,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear Chat", variant="stop")

        # Example questions
        gr.Markdown("### Example Questions")
        gr.Examples(
            examples=[
                ["What is machine learning?"],
                ["Explain the difference between supervised and unsupervised learning."],
                ["What are neural networks?"],
                ["What is Retrieval-Augmented Generation?"],
                ["What are the future trends in AI?"],
            ],
            inputs=msg_input,
        )

        # --------------- Event bindings ---------------
        sample_btn.click(
            fn=load_sample_docs,
            inputs=[mode_selector],
            outputs=[ingest_status],
        )

        upload_btn.click(
            fn=upload_documents,
            inputs=[file_upload, mode_selector],
            outputs=[ingest_status],
        )

        send_btn.click(
            fn=chat,
            inputs=[msg_input, chatbot, mode_selector],
            outputs=[chatbot, sources_display],
        ).then(lambda: "", outputs=[msg_input])

        msg_input.submit(
            fn=chat,
            inputs=[msg_input, chatbot, mode_selector],
            outputs=[chatbot, sources_display],
        ).then(lambda: "", outputs=[msg_input])

        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, sources_display, msg_input],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
