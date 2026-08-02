"""Offline tests for dense FAISS indexing and retrieval."""

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.vector_stores.faiss import FaissVectorStore

from src.ingestion import (
    create_hierarchical_nodes,
    create_hierarchy_docstore,
    load_documents,
    select_leaf_nodes,
)
from src.retrieval import (
    create_dense_index,
    expand_to_parent_contexts,
    print_retrieval_debug,
    retrieve_semantically,
)


class DeterministicEmbedding(BaseEmbedding):
    """Embed interest-rate and AI text in distinct, predictable directions."""

    def _get_query_embedding(self, query: str) -> list[float]:
        """Return the deterministic embedding for a query."""
        return self._embed(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Return the deterministic embedding for an asynchronous query."""
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        """Return the deterministic embedding for document text."""
        return self._embed(text)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Return the deterministic embedding for asynchronous document text."""
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        """Map interest-rate text and AI text to separate fixed vectors."""
        return [1.0, 0.0] if "interest" in text.lower() else [0.0, 1.0]


def test_create_dense_index_embeds_every_leaf_node_in_faiss() -> None:
    """Store one local test vector for each leaf node without an API call."""
    leaf_nodes = select_leaf_nodes(create_hierarchical_nodes(load_documents()))
    vector_index = create_dense_index(
        leaf_nodes,
        embed_model=MockEmbedding(embed_dim=8),
        embedding_dimension=8,
    )

    vector_store = vector_index.vector_store

    assert isinstance(vector_store, FaissVectorStore)
    assert vector_store._faiss_index.ntotal == len(leaf_nodes)


def test_retrieve_semantically_ranks_the_closest_node_first() -> None:
    """Rank an interest-rate node above an unrelated AI node."""
    leaf_nodes = [
        TextNode(
            text="Interest rates affect equity valuations.",
            metadata={"file_name": "rates.txt"},
        ),
        TextNode(
            text="AI spending supports cloud growth.",
            metadata={"file_name": "ai.txt"},
        ),
    ]
    dense_index = create_dense_index(
        leaf_nodes,
        embed_model=DeterministicEmbedding(),
        embedding_dimension=2,
    )

    results = retrieve_semantically("How do interest rates affect equities?", dense_index)

    assert results[0].node.metadata["file_name"] == "rates.txt"
    assert results[0].score is not None
    assert results[1].score is not None
    assert results[0].score < results[1].score


def test_expand_to_parent_contexts_returns_one_context_for_sibling_leaf_matches() -> None:
    """Deduplicate sibling matches while preserving their shared parent context."""
    nodes = create_hierarchical_nodes(load_documents())
    leaf_nodes = select_leaf_nodes(nodes)
    docstore = create_hierarchy_docstore(nodes)
    first_leaf = leaf_nodes[0]
    sibling_leaf = next(
        node
        for node in leaf_nodes[1:]
        if node.parent_node is not None
        and node.parent_node.node_id == first_leaf.parent_node.node_id
    )

    parent_contexts = expand_to_parent_contexts(
        [
            NodeWithScore(node=first_leaf, score=0.1),
            NodeWithScore(node=sibling_leaf, score=0.2),
        ],
        docstore,
    )

    assert len(parent_contexts) == 1
    assert parent_contexts[0].node.node_id == first_leaf.parent_node.node_id
    assert parent_contexts[0].score == 0.1


def test_print_retrieval_debug_includes_score_source_and_preview(capsys: object) -> None:
    """Print inspectable semantic-retrieval details."""
    leaf_nodes = [
        TextNode(
            text="Interest rates affect equity valuations.",
            metadata={"file_name": "rates.txt"},
        )
    ]
    dense_index = create_dense_index(
        leaf_nodes,
        embed_model=DeterministicEmbedding(),
        embedding_dimension=2,
    )
    results = retrieve_semantically("How do interest rates affect equities?", dense_index)

    print_retrieval_debug(results)

    output = capsys.readouterr().out
    assert "FAISS squared L2 distance (lower is closer):" in output
    assert "Source: rates.txt" in output
    assert "Text: Interest rates affect equity valuations." in output
