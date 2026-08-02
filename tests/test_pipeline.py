"""Offline tests for the assembled semantic RAG pipeline."""

from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM

from src.pipeline import run_pipeline


def test_run_pipeline_retrieves_context_and_renders_the_prompt(capsys: object) -> None:
    """Run every semantic RAG stage without OpenAI credentials."""
    question = "What are the interest-rate risks for technology equities?"

    response = run_pipeline(
        question,
        embed_model=MockEmbedding(embed_dim=1_536),
        llm=MockLLM(),
    )

    debug_output = capsys.readouterr().out

    assert "--- SEMANTIC RETRIEVAL DEBUG ---" in debug_output
    assert question in response
    assert "Retrieved context:" in response
