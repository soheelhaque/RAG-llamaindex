"""Assembly point for the LlamaIndex RAG pipeline."""

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI

from src.ingestion import (
    create_hierarchical_nodes,
    create_hierarchy_docstore,
    load_documents,
    select_leaf_nodes,
)
from src.prompts import create_prompt_template, format_parent_contexts
from src.retrieval import (
    create_dense_index,
    expand_to_parent_contexts,
    print_retrieval_debug,
    retrieve_semantically,
)

LLM_MODEL = "gpt-4.1-mini"


def run_pipeline(
    question: str,
    *,
    embed_model: BaseEmbedding | None = None,
    llm: LLM | None = None,
) -> str:
    """Answer a question with semantic leaf retrieval and parent context.

    Args:
        question: The financial research question to answer.
        embed_model: Optional embedding model for the FAISS index. Tests pass a
            local model; the default is the OpenAI embedding model.
        llm: Optional language model used to generate the response. Tests pass
            a local model; the default is OpenAI.

    Returns:
        The generated investment-style response.
    """
    documents = load_documents()
    all_nodes = create_hierarchical_nodes(documents)
    hierarchy_docstore = create_hierarchy_docstore(all_nodes)
    leaf_nodes = select_leaf_nodes(all_nodes)
    dense_index = create_dense_index(leaf_nodes, embed_model=embed_model)

    retrieved_leaf_nodes = retrieve_semantically(question, dense_index)
    print_retrieval_debug(retrieved_leaf_nodes)
    parent_contexts = expand_to_parent_contexts(retrieved_leaf_nodes, hierarchy_docstore)
    context = format_parent_contexts(parent_contexts)
    prompt = create_prompt_template().format(context=context, question=question)

    if llm is None:
        llm = OpenAI(model=LLM_MODEL)

    return llm.complete(prompt).text
