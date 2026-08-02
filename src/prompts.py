"""Prompt construction for retrieved financial context."""

from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore


def format_parent_contexts(parent_contexts: list[NodeWithScore]) -> str:
    """Combine expanded parent contexts into source-labelled prompt context.

    Args:
        parent_contexts: The parent nodes selected from leaf-node retrieval.

    Returns:
        Parent text separated by source labels and divider lines.
    """
    return "\n\n---\n\n".join(
        f"Source: {result.node.metadata.get('file_name', 'unknown source')}\n"
        f"{result.node.get_content()}"
        for result in parent_contexts
    )


def create_prompt_template() -> PromptTemplate:
    """Create the financial-research prompt for expanded retrieval context.

    Returns:
        A template that accepts ``context`` and ``question`` values.
    """
    return PromptTemplate(
        """You are a financial research assistant. Answer the question using only the
provided context. If the context does not contain enough information, say so clearly.

Retrieved context:
{context}

Question:
{question}

Provide a concise investment-style response with 5 to 10 bullet points. Add a short
"Risk notes" section when relevant."""
    )
