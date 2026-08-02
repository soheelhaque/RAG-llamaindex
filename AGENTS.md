# RAG-llamaindex

## Purpose

This project is a small LlamaIndex implementation of the Financial Research Assistant defined in
`docs/SPEC.md`. Its purpose is to understand document-oriented RAG concepts, including hierarchical
chunking and hybrid dense/sparse retrieval.

## Scope

- Keep the implementation small and focused.
- Follow the architecture in `docs/SPEC.md`.
- Do not add production features, a user interface, APIs, agents, memory, evaluation frameworks,
  reranking, or advanced query strategies unless explicitly requested.
- Use existing dependencies before adding new ones.

## Conventions

- Use Python 3.13.9 and uv.
- Use Ruff with a 100-character line length.
- Prefer straightforward, readable code over abstractions.
- Keep the four implementation modules focused on their named responsibilities.
- Build the implementation incrementally so that each LlamaIndex concept remains inspectable.
