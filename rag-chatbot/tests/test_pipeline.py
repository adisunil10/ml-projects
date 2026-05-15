"""Integration tests for the RAGPipeline."""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import RAGPipeline


SAMPLE_DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


@pytest.fixture(scope="module")
def pipeline():
    """Pipeline pre-loaded with sample documents."""
    p = RAGPipeline(mode="local")
    p.ingest(SAMPLE_DOCS_DIR)
    return p


class TestRAGPipeline:
    def test_ingest_indexes_vectors(self, pipeline):
        assert len(pipeline.vector_store) > 0

    def test_query_returns_answer(self, pipeline):
        result = pipeline.query("What is machine learning?")
        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_query_returns_sources(self, pipeline):
        result = pipeline.query("What are neural networks?")
        assert "sources" in result
        assert isinstance(result["sources"], list)
        assert len(result["sources"]) > 0

    def test_sources_have_score(self, pipeline):
        result = pipeline.query("Explain deep learning.")
        for source in result["sources"]:
            assert "score" in source
            assert isinstance(source["score"], float)
            assert 0.0 <= source["score"] <= 1.01  # allow tiny float tolerance

    def test_query_returns_scores_dict(self, pipeline):
        result = pipeline.query("What is AI?")
        assert "scores" in result
        scores = result["scores"]
        for key in ("context_relevance", "answer_faithfulness", "answer_relevancy", "overall"):
            assert key in scores
            assert 0.0 <= scores[key] <= 1.0

    def test_k_parameter_limits_sources(self):
        p = RAGPipeline(mode="local", k=2)
        p.ingest(SAMPLE_DOCS_DIR)
        result = p.query("What is supervised learning?")
        assert len(result["sources"]) <= 2

    def test_save_and_load_index(self, pipeline):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test_index"
            pipeline.save_index(index_path)

            new_pipeline = RAGPipeline(mode="local")
            new_pipeline.load_index(index_path)

            assert len(new_pipeline.vector_store) == len(pipeline.vector_store)

            result = new_pipeline.query("What is AI?")
            assert len(result["answer"]) > 0

    def test_empty_pipeline_no_crash(self):
        p = RAGPipeline(mode="local")
        result = p.query("anything")
        assert "answer" in result
        assert "No documents" in result["answer"]

    def test_incremental_ingest(self):
        p = RAGPipeline(mode="local")
        sample_file = SAMPLE_DOCS_DIR / "sample.txt"
        n = p.add_documents_from_files([sample_file])
        assert n > 0
        assert len(p.vector_store) == n
