Query Pipeline Implementation Plan

This document outlines the implementation steps for adding query support to the RAG system. The goal is to start with a simple CLI query command while keeping the architecture reusable for future consumers such as MCP integrations or other programmatic callers.

⸻

1. Define query domain models

Goal

Create a small set of typed request/response models for querying.

Requirements

Add models for:
	•	QueryRequest
	•	RetrievedChunk
	•	QueryResponse

QueryRequest should include:
	•	query text
	•	top_k
	•	optional generation flag
	•	optional query/generation model overrides

RetrievedChunk should include:
	•	chunk_id
	•	doc_id
	•	text
	•	score
	•	chunk_index
	•	optional metadata (offsets, heading path, page)

QueryResponse should include:
	•	original query
	•	retrieved matches
	•	optional generated answer
	•	info about models used

Notes

These models are the stable contract between core query logic and frontends. CLI should format them, not define them.

⸻

2. Add query module structure

Goal

Create a dedicated core query package.

Requirements

Add:

raggen/core/query/
  __init__.py
  models.py
  service.py
  retriever.py
  generator.py
  context_builder.py

Only retrieval needs to be implemented first. generator.py and context_builder.py can be stubs.

Notes

Keep query logic out of CLI code. CLI should call into query.service.

⸻

3. Extend vector backend interface with similarity search

Goal

Allow each vector backend to perform nearest-neighbor search.

Requirements

Extend VectorBackend with a method like:

def search(self, engine, *, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
    ...

Return:
	•	chunk_id
	•	similarity/distance score

Implement for:
	•	sqlite-vec backend
	•	pgvector backend

Notes

Do not implement search directly in query service. Keep all backend-specific SQL inside the backend classes.

⸻

4. Add metadata fetch by chunk ids

Goal

Fetch full chunk rows for retrieved chunk ids.

Requirements

Add metadata store method:

def fetch_chunks_by_ids(engine, chunk_ids: list[str]) -> list[dict]:
    ...

Return enough data to build RetrievedChunk.

Should include:
	•	chunk text
	•	doc_id
	•	chunk_index
	•	offsets
	•	heading/page metadata if available

Notes

Order should match retrieval order, not database default order.

⸻

5. Implement retrieval service

Goal

Create the first working core query flow.

Requirements

Implement in raggen/core/query/service.py:

def query(request: QueryRequest, runtime=None) -> QueryResponse:
    ...

Flow:
	1.	load config/runtime
	2.	validate query embedding configuration
	3.	embed query text
	4.	call vector backend search(...)
	5.	fetch chunk rows from metadata store
	6.	build QueryResponse

First implementation should be retrieval-only. answer can be None.

Notes

Do not print from this function. Return structured data only.

⸻

6. Add query embedding support

Goal

Support embedding the query text consistently with stored vectors.

Requirements

Use the configured query embedding model.

Validate:
	•	query embedding dimension matches stored embedding dimension
	•	if query model config is empty, fall back to ingestion embedding model

Notes

Do not assume query embedding model and generation LLM are the same thing. Keep them separate in config and logic.

⸻

7. Add configuration stubs for querying and generation

Goal

Prepare config for future query/generation extensibility.

Requirements

Extend config with sections like:

[query]
model_id = ""
top_k = 8

[generation]
enabled = false
provider = ""
model_id = ""

At minimum:
	•	query model id
	•	default top_k
	•	generation enabled flag
	•	generation model id placeholder

Notes

Generation does not need implementation yet, but config shape should not block it later.

⸻

8. Add CLI query command

Goal

Expose retrieval through a simple CLI command.

Requirements

Add:

rag query "how does bootstrap work?"

CLI should:
	1.	bootstrap/load runtime
	2.	build QueryRequest
	3.	call core query service
	4.	print matches in a readable way

Output should include:
	•	document path
	•	score
	•	short text snippet

Notes

Keep CLI rendering thin. No retrieval logic in the command itself.

⸻

9. Add generation stub layer

Goal

Reserve a clean extension point for answer synthesis.

Requirements

Create generator.py with a placeholder interface, such as:

def generate_answer(query: str, chunks: list[RetrievedChunk], model_id: str) -> str:
    raise NotImplementedError

Query service can call this only if generation is enabled, but initial implementation may simply skip generation.

Notes

This keeps future LLM provider support isolated from retrieval.

⸻

10. Add tests for retrieval flow

Goal

Verify query pipeline works end-to-end.

Requirements

Add tests for:
	•	vector backend search(...)
	•	metadata fetch by chunk ids
	•	query service returns ordered results
	•	CLI query command prints expected output

Use small deterministic test data and avoid network/model downloads where possible.

Notes

Prefer a fake/dummy embedding implementation for query tests.

⸻

11. Validate runtime/config consistency

Goal

Prevent broken query behavior from mismatched config.

Requirements

Before running retrieval, validate:
	•	runtime is initialized
	•	engine is available
	•	vector backend is available
	•	query embedding dimension matches stored embedding dimension

Raise clear errors if not.

Notes

This will save a lot of debugging when users change config manually or use the wrong project.

⸻

12. Keep retrieval and generation separate

Goal

Preserve clean architecture for future growth.

Requirements

Ensure:
	•	retrieval works independently
	•	generation is optional
	•	query response always contains retrieved evidence, even if generation is added later

Notes

This is important for trust, debugging, and MCP-style consumers.

⸻

Suggested implementation order
	1.	Query models
	2.	Query module structure
	3.	Vector backend search()
	4.	Metadata fetch by chunk ids
	5.	Retrieval service
	6.	Query embedding support
	7.	Config query/generation stubs
	8.	CLI query command
	9.	Generation stub
	10.	Tests
	11.	Runtime/config validation
	12.	Cleanup and polish

⸻

First milestone

The first usable milestone is:
	•	rag query "..." works
	•	retrieval returns top-k matching chunks
	•	results are printed in CLI
	•	no answer generation yet

That is enough to validate the end-to-end query architecture before adding LLM synthesis later.
