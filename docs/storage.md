# Storage Layer (Metadata + Pluggable Vector Backends)

This module implements the **database layer** of the RAG system.

It is responsible for:

- Persisting documents, chunks, and embedding metadata
- Managing project configuration (model, dimension, backend)
- Delegating vector storage to a pluggable backend
- Supporting multiple databases via SQLAlchemy

It is intentionally designed to be:
- **Portable**
- **Extensible**
- **Deterministic**
- **Backend-agnostic for metadata**

---

## Design Philosophy

The storage layer is split into two clear parts:

### 1 Metadata Layer (Portable)

Managed with **SQLAlchemy** and works with any database that SQLAlchemy supports.

Stores:
- `rag_project` (init configuration)
- `documents`
- `chunks`
- `embeddings` (metadata only)

This layer contains the conceptual truth of the system:
- What documents exist
- How they were chunked
- Which embedding model/dimension was used

It does **not** know how vector similarity works.

---

### 2 Vector Backend (Pluggable)

Vector storage is implemented via a **backend plugin**.

Built-in backends:
- `sqlite_vec`
- `pgvector`

Each backend handles:
- Creating vector schema
- Dropping vector schema (for destructive re-init)
- Inserting/upserting vectors

This separation keeps the system clean:
- Metadata is portable.
- Vector mechanics are isolated.

---

## Initialization

Initialization locks the storage contract.

```bash
rag init \
  --database-url sqlite:///./.rag/rag.db \
  --backend-key sqlite_vec \
  --vector-backend-import raggen.core.store.vector_backends.sqlite_vec:SQLiteVecBackend \
  --embedding-dim 384 \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2

During init:
	1.	SQLAlchemy engine is created.
	2.	Metadata tables are created.
	3.	Vector backend schema is created.
	4.	A single rag_project row is written.

If configuration changes later, you must re-initialize with --destructive.

⸻

How It Works (High-Level)

During ingestion:
	1.	Document metadata is stored.
	2.	Chunk rows are stored.
	3.	Embedding metadata is stored.
	4.	Vectors are inserted via the vector backend.

All operations are transactional.
If vector insertion fails, nothing is committed.

The metadata schema remains stable regardless of backend.

⸻

Why This Architecture
	•	SQLAlchemy provides broad database compatibility.
	•	Vector backends differ significantly across databases.
	•	Separating metadata from vectors keeps portability high.
	•	Init-time config enforces consistency and reproducibility.
	•	Destructive re-init simplifies schema evolution.

This avoids tightly coupling the system to a single vector database.

⸻

Extending the Vector Store

You can implement your own vector backend without modifying core code.

Step 1: Create a Backend Class

from raggen.core.store.vector_backends.base import VectorBackend

class MyBackend(VectorBackend):
    key = "my_backend"

    def supports(self, engine):
        return engine.dialect.name == "mydb"

    def create_schema(self, engine, dim):
        ...

    def drop_schema(self, engine):
        ...

    def upsert_vectors(self, engine, *, vectors, embedding_model_id, dim, normalized):
        ...

Step 2: Point Config to It

--vector-backend-import mypackage.module:MyBackend

As long as:
	•	SQLAlchemy can connect to the database
	•	Your backend implements the required interface

⸻

Supported Backends (Built-in)

SQLite + sqlite-vec
	•	Local development
	•	Lightweight
	•	Embeddings stored in a virtual table

Postgres + pgvector
	•	Production-ready
	•	Vector type + extension
	•	Clean upsert support

⸻

What This Layer Does Not Do
	•	It does not perform similarity search (retrieval).
	•	It does not manage migrations between configs.
	•	It does not hide database-specific vector limitations.

It focuses strictly on stable persistence.

⸻

Summary
	•	Metadata is portable.
	•	Vector storage is pluggable.
	•	Init locks invariants.
	•	Extensions require only a small backend class.
	•	The architecture scales from local SQLite to production Postgres.

This storage layer is designed to be simple to use and powerful to extend.
