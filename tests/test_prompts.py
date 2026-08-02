"""Offline tests for financial-research prompt construction."""

from llama_index.core.schema import NodeWithScore, TextNode

from src.prompts import create_prompt_template, format_parent_contexts


def test_format_parent_contexts_includes_source_labels_and_text() -> None:
    """Keep the context traceable to its research-note source."""
    parent_contexts = [
        NodeWithScore(
            node=TextNode(
                text="Interest rates can pressure equity valuations.",
                metadata={"file_name": "rates.txt"},
            ),
            score=0.1,
        )
    ]

    context = format_parent_contexts(parent_contexts)

    assert context == "Source: rates.txt\nInterest rates can pressure equity valuations."


def test_create_prompt_template_inserts_context_and_question() -> None:
    """Render the values that will be supplied to the LLM later."""
    prompt = create_prompt_template()

    rendered_prompt = prompt.format(
        context="Source: rates.txt\nInterest rates can pressure equity valuations.",
        question="What are the valuation risks?",
    )

    assert "Source: rates.txt" in rendered_prompt
    assert "What are the valuation risks?" in rendered_prompt
    assert "5 to 10 bullet points" in rendered_prompt
