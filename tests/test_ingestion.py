"""Offline tests for loading the financial corpus."""

from pathlib import Path

from llama_index.core.node_parser import get_root_nodes
from llama_index.core.schema import NodeRelationship

from src.ingestion import (
    CHILD_CHUNK_SIZE,
    CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    create_hierarchical_nodes,
    create_hierarchy_docstore,
    load_documents,
    select_leaf_nodes,
)


def test_load_documents_loads_the_financial_corpus() -> None:
    """Load every expected note with its LlamaIndex file metadata."""
    documents = load_documents()

    assert len(documents) == 8
    assert all(document.text for document in documents)
    assert all("file_name" in document.metadata for document in documents)
    assert all("file_path" in document.metadata for document in documents)


def test_load_documents_orders_documents_by_file_name() -> None:
    """Provide a stable order for demos and tests."""
    documents = load_documents()

    file_names = [document.metadata["file_name"] for document in documents]
    source_names = [Path(document.metadata["file_path"]).name for document in documents]

    assert file_names == sorted(file_names)
    assert file_names == source_names


def test_create_hierarchical_nodes_creates_parent_and_child_nodes() -> None:
    """Create a two-level hierarchy that retains the source file metadata."""
    nodes = create_hierarchical_nodes(load_documents())
    root_nodes = get_root_nodes(nodes)
    leaf_nodes = select_leaf_nodes(nodes)

    assert PARENT_CHUNK_SIZE == 1_000
    assert CHILD_CHUNK_SIZE == 300
    assert CHUNK_OVERLAP == 50
    assert len(root_nodes) == 8
    assert leaf_nodes
    assert len(nodes) > len(root_nodes)
    assert all("file_name" in node.metadata for node in nodes)
    assert all(NodeRelationship.PARENT in node.relationships for node in leaf_nodes)


def test_select_leaf_nodes_excludes_parent_nodes() -> None:
    """Keep broader parent chunks out of later retrieval indexes."""
    nodes = create_hierarchical_nodes(load_documents())
    leaf_nodes = select_leaf_nodes(nodes)
    root_nodes = get_root_nodes(nodes)

    assert leaf_nodes
    assert not {node.node_id for node in leaf_nodes}.intersection(
        node.node_id for node in root_nodes
    )


def test_create_hierarchy_docstore_resolves_a_leaf_nodes_parent() -> None:
    """Retain parent chunks so precise retrieval can later expand its context."""
    nodes = create_hierarchical_nodes(load_documents())
    leaf_node = select_leaf_nodes(nodes)[0]
    docstore = create_hierarchy_docstore(nodes)

    parent_node = docstore.get_document(leaf_node.parent_node.node_id)

    assert parent_node is not None
    assert parent_node.node_id == leaf_node.parent_node.node_id
    assert parent_node.metadata["file_name"] == leaf_node.metadata["file_name"]
