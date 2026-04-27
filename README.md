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

- Build the database:

  rag build

- Ingest files from the current directory:

  rag ingest

- Query the index:

  rag query "what does the config system do?"

Vector backends

raggen ships with two vector backends out of the box:

- sqlite_vec  local SQLite database using the sqlite-vec extension.
              Good default for single-machine use. No extra infrastructure needed.

- pgvector    PostgreSQL with the pgvector extension.
              Good for shared or production deployments.

sqlite_vec works out of the box — no extra dependencies needed. For pgvector,
install the PostgreSQL driver first:

  pip install raggen[postgres]

Select a backend in .rag/config.toml:

  [storage]
  backend_key  = "sqlite_vec"
  database_url = "sqlite:///.rag/rag.db"

  or

  [storage]
  backend_key  = "pgvector"
  database_url = "postgresql://user:pass@localhost/mydb"

Custom vector backends

You can bring your own vector store (Qdrant, Chroma, Weaviate, a proprietary
system, etc.) by implementing the VectorBackend interface and pointing the
config at your class:

  [storage]
  backend_key           = "my_backend"
  vector_backend_import = "my_package.backends:MyVectorBackend"

raggen will import and instantiate your class at runtime. The interface
requires six methods: supports, create_schema, drop_schema, upsert_vectors,
delete_vectors, and search.

The full contract — including the transaction model, score conventions, and
implementation notes — is documented in:

  src/raggen/core/store/vector_backends/base.py

Important: database_url is not passed to your backend. raggen uses it only
to connect to the metadata database (see below). Your backend is responsible
for its own connection — read a URL from an environment variable, a separate
config file, or hardcode it for local development. The Engine and Connection
objects your methods receive belong to the metadata database and should not
be used to talk to your vector store.

Metadata storage

Document metadata (file paths, chunk text, embeddings) is always stored via
SQLAlchemy using the database_url from [storage]. Any database supported by
SQLAlchemy works here without any additional code.

For the built-in backends (sqlite_vec, pgvector) the vector data lives in
the same database as the metadata, so database_url covers both. For a custom
backend that connects to a separate service, database_url covers metadata
only.

License
- MIT (see LICENSE)
