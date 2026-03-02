Design Doc: RAG Bootstrapper (Core-First)

Scope (for this phase)
	•	Python-only engine that can:
	1.	scan a directory
	2.	parse + chunk (reuse your existing code)
	3.	embed (stub first, real embedder later)
	4.	index + retrieve (storage adapter)
	•	Everything else (CLI/API) is a wrapper around the engine.

Non-goals (for now)
	•	Generation / LLM answering
	•	Observability dashboards
	•	Multi-tenant, auth, remote hosting

⸻

0. Core principles / invariants (lock these first)

0.1 Determinism
	•	Same file contents + same config => same chunk ids and same stored records (upsert, no duplicates).

0.2 Clear boundaries
	•	Core has no CLI concerns and no FastAPI concerns.
	•	Storage is a “port” interface (adapter pattern).

0.3 Versioning
	•	Parser version, chunk config hash, embedder id/version are stored with records.
	•	If any changes: tool can detect “reindex required”.

Tests
	•	Golden test for chunk id stability (same input -> same output).
	•	“Upsert twice” should not create duplicates.

⸻

1. Core module layout (library-first)

Goal: define structure so everything plugs into it cleanly.

1.1 Packages
	•	rag_core/
	•	models.py (Pydantic/dataclasses)
	•	scanner.py
	•	pipeline.py (indexing orchestration)
	•	retrieval.py
	•	ids.py (hashing, id generation)
	•	ports.py (interfaces)
	•	rag_adapters/
	•	store_sqlite.py (later)
	•	embedder_local.py (later)
	•	embedder_stub.py (now)
	•	tests/

Tests
	•	Import-level: package imports work, no cyclic deps.
	•	Static typing smoke (optional, but nice).

⸻

2. Core contracts (models + ports)

Goal: define minimal data shapes that won’t change every week.

2.1 Models (minimum)
	•	FileRef:
	•	path, rel_path, size, mtime, content_hash, mime/type
	•	IndexConfig:
	•	wraps your ChunkConfig
	•	embedding config (provider, model, dim, batch size)
	•	ignore config
	•	DocumentRecord:
	•	doc_id, rel_path, content_hash, parser_version
	•	ChunkRecord:
	•	chunk_id, doc_id, text, offset_start, offset_end, metadata, config_hash
	•	VectorRecord:
	•	chunk_id, vector (list[float] or bytes), dim
	•	SearchResult:
	•	chunk_id, score, plus hydrated ChunkRecord fields (or a reference)

2.2 Ports (interfaces)
	•	ParserPort: parse(file_path) -> Document
	•	ChunkerPort: chunk(document, chunk_config) -> list[ChunkRecord-like]
	•	EmbedderPort:
	•	embed_texts(list[str]) -> list[Vector]
	•	embed_query(str) -> Vector
	•	model_id, dim
	•	VectorStorePort:
	•	upsert_documents(docs)
	•	upsert_chunks(chunks)
	•	upsert_vectors(vectors)
	•	delete_document(doc_id) (or mark inactive)
	•	query(vector, top_k, filters?) -> list[(chunk_id, score)]
	•	get_chunks(chunk_ids) -> list[ChunkRecord]
	•	StateStorePort (optional but useful):
	•	track runs / last indexed time (can also live in vector store schema)

Tests
	•	Interface compliance tests via stub adapters (fake store + fake embedder).
	•	Pydantic validation tests for config edge cases.

⸻

3. Stable identity + hashing (critical path)

Goal: define IDs before writing storage.

3.1 doc_id
	•	Deterministic from rel_path + content_hash (or just rel_path and store content_hash separately).
	•	Recommendation: keep doc_id stable per path; use content_hash to detect changes.

3.2 chunk_id

Deterministic hash of:
	•	doc_id
	•	offset_start, offset_end
	•	config_hash (your stable chunk config hash)
	•	(optional) chunker_version

3.3 config_hash
	•	Reuse your _stable_config_hash(conf).

Tests
	•	Changing chunk size changes chunk ids predictably.
	•	Same input text produces same offsets and ids.
	•	Repeated substring cases don’t break offsets (you already have logic—test it here).

⸻

4. Scanner (filesystem → FileRef list)

Goal: scan a directory reproducibly and ignore the right stuff.

4.1 Ignore rules
	•	Read .gitignore
	•	Read optional .ragignore
	•	Default ignores: .git/, .rag/, node_modules/, dist/, binaries

4.2 File typing
	•	By extension + simple mime sniff (keep simple)
	•	For now: text-ish files only (md, txt, py, ts, js, go, json, yaml, etc.)

4.3 Hashing
	•	Compute content_hash (sha256 of bytes) for deterministic incremental indexing.

Tests
	•	Scanner respects ignore rules.
	•	Scanner output stable ordering (sort by rel_path).
	•	Hash changes when file changes.

⸻

5. Index planning (incremental)

Goal: decide what to parse/embed without doing work unnecessarily.

5.1 Planner behavior

Given current scan + stored documents:
	•	New files => index
	•	Changed hash => re-index
	•	Missing files => delete/mark inactive

Tests
	•	Planner returns correct sets (new/changed/deleted) for a temp directory fixture.
	•	Idempotence: second run on same dir yields no work.

⸻

6. Index pipeline (parse → chunk → embed → store)

Goal: implement the orchestration logic in core, using ports.

6.1 Pipeline function (core)

index_directory(root, config, ports) -> IndexReport
	•	Scan
	•	Plan
	•	For each file to index:
	•	Parse (reuse your parser)
	•	Chunk (reuse your chunker)
	•	Enrich chunk metadata:
	•	rel_path, doc_type, pagenumber if present, offsets, config_hash
	•	Embed chunk texts (batched)
	•	Upsert: docs, chunks, vectors
	•	Handle deletes

6.2 Reporting
	•	Return counts + timings:
	•	files scanned, indexed, skipped, deleted
	•	chunks created
	•	embeddings created

Tests
	•	End-to-end core test with:
	•	stub embedder (returns deterministic vectors)
	•	in-memory fake store (dict-backed)
	•	asserts on counts and stored records
	•	Re-indexing unchanged dir produces zero new chunks/vectors.

⸻

7. Retrieval (query → top-k chunks)

Goal: get search working before “real” DB.

7.1 Retrieval function (core)

retrieve(query_text, top_k, filters?) -> list[SearchResult]
	•	Embed query
	•	Store.query
	•	Store.get_chunks
	•	Assemble results

7.2 Filters (optional now, design for later)
	•	path prefix
	•	file type
	•	doc_id

Tests
	•	With fake store:
	•	known vectors return expected nearest chunks
	•	With deterministic embedder:
	•	query gets stable results

⸻

8. First real adapters (after core is solid)

8A. EmbedderStub (NOW)

Goal: unlock end-to-end tests without model dependencies.
	•	Deterministic vector from hashing text (fixed dim)

Tests
	•	Same text => same vector
	•	dim matches config

8B. SQLite Vector Store (NEXT)

Goal: persist locally with minimal deps.
	•	Tables:
	•	documents
	•	chunks
	•	vectors
	•	Vector search:
	•	either sqlite vector extension OR fallback brute force for small corpora (fine for MVP)

Tests
	•	Upsert then query returns expected
	•	Re-index updates changed docs
	•	Delete removes chunks/vectors

(Once this works, Postgres+pgvector becomes a backend swap.)

⸻

9. Thin wrappers (only after core + adapters)

9.1 CLI (Typer)

Commands map 1:1 to core:
	•	rag init → write config + init db
	•	rag index → call index_directory
	•	rag query → call retrieve
	•	rag serve → optional (starts API)

Tests
	•	CLI smoke tests (subprocess) optional
	•	Most logic stays in core tests

9.2 API (FastAPI)
	•	POST /retrieve → core retrieve
	•	POST /index → core index (optional)
	•	GET /health

Tests
	•	API contract tests with FastAPI test client

⸻

Suggested implementation order (the TODO list)
	1.	Core contracts + ports (models.py, ports.py) + tests
	2.	IDs + hashing (ids.py) + golden tests
	3.	Scanner (scanner.py) + ignore/hash tests
	4.	Planner (planner.py or in pipeline.py) + incremental tests
	5.	Index pipeline (pipeline.py) using:
	•	your Parser + Chunker
	•	EmbedderStub
	•	FakeStore (dict-backed)
	6.	Retrieval core (retrieval.py) + tests with fake store
	7.	SQLite store adapter + integration tests
	8.	CLI (Typer) thin wrapper + smoke tests
	9.	FastAPI optional wrapper + contract tests

⸻

Done when (for “core complete”)
	•	You can run a test that:
	•	creates a temp directory with a few files
	•	indexes them (parse→chunk→embed→store)
	•	queries and returns top-k chunks
	•	re-runs indexing with no changes and does no work
	•	changes one file and only re-indexes that file

