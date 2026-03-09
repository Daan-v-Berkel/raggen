# raggen

Local-first RAG (retrieval-augmented generation) pipeline library and CLI.

Lightweight toolkit for scanning a project tree, parsing files, chunking text, and storing/embedding chunks with pluggable vector backends.

Requirements
- Python 3.12+
- See pyproject.toml for runtime dependencies

Quickstart

1. create and activate venv

   python -m venv .venv && source .venv/bin/activate

2. Install editable package for development:

   pip install -e .

3. Run tests:

   pytest -q

4. Use the CLI (after installation):

   rag -h

Common workflows

- Initialize a project (creates .rag/config.toml):

  rag init

- Ingest/initialize DB:

  rag ingest

License
- MIT (see LICENSE)
