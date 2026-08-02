"""Dense, sparse, and hybrid retrieval helpers."""

import faiss
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1_536


def create_dense_index(
    leaf_nodes: list[BaseNode],
    embed_model: BaseEmbedding | None = None,
    embedding_dimension: int = EMBEDDING_DIMENSION,
) -> VectorStoreIndex:
    """Embed leaf nodes and store their vectors in an in-memory FAISS index.

    Args:
        leaf_nodes: The smallest hierarchical nodes to make retrievable.
        embed_model: Embedding model to use. Defaults to OpenAI's small
            embedding model; tests can supply a local model instead.
        embedding_dimension: Vector dimension expected by the embedding model.

    Returns:
        A LlamaIndex vector index backed by FAISS.
    """
    if embed_model is None:
        embed_model = OpenAIEmbedding(
            model=EMBEDDING_MODEL,
            dimensions=embedding_dimension,
        )

    vector_store = FaissVectorStore(faiss.IndexFlatL2(embedding_dimension))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )


def retrieve_semantically(
    query: str,
    dense_index: VectorStoreIndex,
    k: int = 4,
) -> list[NodeWithScore]:
    """Retrieve the leaf nodes nearest to a query in the FAISS index.

    Args:
        query: The financial research question to embed and search for.
        dense_index: The FAISS-backed index containing leaf-node embeddings.
        k: Maximum number of matching nodes to return.

    Returns:
        The top matching nodes with FAISS squared L2 distances. Lower scores
        indicate closer matches.
    """
    retriever = dense_index.as_retriever(similarity_top_k=k)
    return retriever.retrieve(query)


def expand_to_parent_contexts(
    retrieved_leaf_nodes: list[NodeWithScore],
    hierarchy_docstore: BaseDocumentStore,
) -> list[NodeWithScore]:
    """Replace retrieved leaf nodes with their broader parent contexts.

    The returned score belongs to the leaf node that caused its parent to be
    selected. When multiple retrieved leaves share a parent, the first (and
    therefore closest) result determines that parent's position and score.

    Args:
        retrieved_leaf_nodes: The ranked leaf nodes returned by retrieval.
        hierarchy_docstore: Store containing the complete parent/child hierarchy.

    Returns:
        Unique parent contexts in the retrieved leaf-node order.
    """
    parent_contexts = []
    selected_parent_ids = set()

    for result in retrieved_leaf_nodes:
        parent = result.node.parent_node
        parent_id = parent.node_id if parent is not None else result.node.node_id
        if parent_id in selected_parent_ids:
            continue

        parent_node = hierarchy_docstore.get_document(parent_id)
        if parent_node is None:
            raise ValueError(
                f"Parent node {parent_id} is missing from the hierarchy document store."
            )

        selected_parent_ids.add(parent_id)
        parent_contexts.append(NodeWithScore(node=parent_node, score=result.score))

    return parent_contexts


def create_chunk_preview(node: BaseNode) -> str:
    """Create a compact beginning-and-end preview of a retrieved node.

    Args:
        node: The retrieved node to summarise.

    Returns:
        The full text for short nodes, otherwise the first and last five words.
    """
    words = node.get_content().split()
    if len(words) <= 10:
        return " ".join(words)
    return f"{' '.join(words[:5])} ... {' '.join(words[-5:])}"


def print_retrieval_debug(retrieved_nodes: list[NodeWithScore]) -> None:
    """Print dense-retrieval details for inspection during the demo.

    Args:
        retrieved_nodes: The scored nodes returned by semantic retrieval.
    """
    print("\n--- SEMANTIC RETRIEVAL DEBUG ---")
    for rank, result in enumerate(retrieved_nodes, start=1):
        node = result.node
        score = result.score if result.score is not None else 0.0
        print(f"\nRank {rank}")
        print(f"FAISS squared L2 distance (lower is closer): {score:.4f}")
        print(f"Source: {node.metadata.get('file_name', 'unknown source')}")
        print(f"Text: {create_chunk_preview(node)}")
    print()
