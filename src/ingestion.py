"""Document loading and hierarchical node construction."""

from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import HierarchicalNodeParser, TokenTextSplitter, get_leaf_nodes
from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore

DATA_DIRECTORY = Path("data/financial_docs")
PARENT_CHUNK_SIZE = 1_000
CHILD_CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


def load_documents(data_directory: Path = DATA_DIRECTORY) -> list[Document]:
    """Load the financial corpus as LlamaIndex documents.

    Args:
        data_directory: Directory containing the source ``.txt`` files.

    Returns:
        The corpus documents, ordered by source filename. Each document retains
        the file metadata added by LlamaIndex's directory reader.
    """
    reader = SimpleDirectoryReader(input_dir=str(data_directory), required_exts=[".txt"])
    documents = reader.load_data()
    return sorted(documents, key=lambda document: str(document.metadata["file_name"]))


def create_hierarchical_nodes(documents: list[Document]) -> list[BaseNode]:
    """Split documents into large parent nodes and smaller child nodes.

    Args:
        documents: The source documents to split.

    Returns:
        A flat list of nodes. Parent nodes provide broader context and child
        nodes include a reference to the parent they were created from.
    """
    node_parser = HierarchicalNodeParser.from_defaults(
        node_parser_ids=["parent", "child"],
        node_parser_map={
            "parent": TokenTextSplitter(
                chunk_size=PARENT_CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            ),
            "child": TokenTextSplitter(
                chunk_size=CHILD_CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            ),
        },
    )
    return node_parser.get_nodes_from_documents(documents)


def select_leaf_nodes(nodes: list[BaseNode]) -> list[BaseNode]:
    """Return only the smallest nodes from a hierarchical node list.

    Args:
        nodes: The flat list containing both parent and child nodes.

    Returns:
        The leaf nodes to embed and search in the dense and sparse indexes.
    """
    return get_leaf_nodes(nodes)


def create_hierarchy_docstore(nodes: list[BaseNode]) -> SimpleDocumentStore:
    """Store every hierarchical node for later parent-context lookup.

    Args:
        nodes: The full hierarchy, including both parent and child nodes.

    Returns:
        An in-memory LlamaIndex document store keyed by node ID.
    """
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    return docstore
