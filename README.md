# RAG-llamaindex

## Project overview

This repository is a small, educational LlamaIndex implementation of the Financial Research
Assistant defined in [`docs/SPEC.md`](docs/SPEC.md). It will explore a document-oriented RAG
architecture with hierarchical chunks, dense FAISS retrieval, sparse BM25 retrieval, and a simple
hybrid retrieval step.

The implementation will be built incrementally to keep each framework concept visible and easy to
experiment with. It is not intended to be a production RAG system.

## Setup

This project uses Python 3.13.9 and uv. The virtual environment has already been created.

```bash
uv sync
```

## Development checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The RAG implementation and runnable demo will be added alongside the learning exercises.
