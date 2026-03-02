# RAG Bootstrapper

A local-first CLI tool that turns any directory into a working Retrieval-Augmented Generation (RAG) pipeline.

Lean.
Python-native.
Built for developers and agents.

---

## Purpose

Initialize RAG over a folder — fast.

This tool bootstraps a complete RAG pipeline directly inside a project directory:

- Parses files
- Chunks content
- Embeds into a vector store
- Exposes retrieval via CLI
- Optionally runs a lightweight local API

Designed primarily for:

- Codebases
- Documentation
- Contracts
- Research
- Mixed project knowledge

---

## Architecture Philosophy

The entire system is Python-based.

No Node.js layer.
No separate orchestration service.

Everything lives in:

- A Python RAG engine
- A Python CLI interface
- A local vector database

The CLI is the primary interface.

A local API is optional and secondary — mainly for agent or MCP integration.

Future web UI support should be possible, but is not the focus.

---

## Core Workflow

Inside any directory:

```bash
rag init
rag index
rag query "How does authentication work?"

That’s it.

⸻

Commands

Initialize

rag init

Interactive setup:
	•	Document type (code, docs, contracts, research, mixed)
	•	Chunk size + overlap
	•	Chunking strategy
	•	Embedding model:
	•	Local (e.g. sentence-transformers, Ollama)
	•	API-based
	•	Query model (optional)
	•	Storage backend
	•	Ignore rules (.gitignore + optional .ragignore)

Creates:

.rag/
  config.yaml
  metadata.db (or Postgres config)


⸻

Index

rag index

Pipeline:
	1.	Parse files
	2.	Normalize content
	3.	Chunk
	4.	Embed
	5.	Store vectors

Features:
	•	Respects .gitignore
	•	Skips binaries
	•	Incremental re-indexing
	•	Content hashing for change detection

⸻

Query

rag query "Where is token validation implemented?"

Returns:
	•	Top-k retrieved chunks
	•	Scores
	•	Optional generated answer
	•	Raw retrieval mode (for agent use)

⸻

Optional API Mode

rag serve

Starts a lightweight FastAPI server:

POST /retrieve
POST /query

Use cases:
	•	Coding agents
	•	MCP tools
	•	IDE integrations
	•	Automation workflows

However, agents can also simply execute:

rag query "..."

from the project root.

The CLI remains the primary interface.

⸻

Core Components

Python RAG Engine
	•	File parsing
	•	Chunking strategies
	•	Embedding pipeline
	•	Vector indexing
	•	Retrieval logic

CLI Layer (Python)
	•	Interactive setup
	•	Command orchestration
	•	Local execution interface

Storage
	•	Default: Postgres + pgvector
	•	Optional: SQLite + local vector store
	•	Fully local by default

⸻

Design Principles

Local-first
	•	Runs entirely on developer machine
	•	Supports fully offline mode

Minimal infrastructure
	•	No required external services
	•	Docker optional, not mandatory

Configurable
	•	Pluggable chunkers
	•	Pluggable embedders
	•	Pluggable retrievers

Incremental
	•	Detect file changes
	•	Re-embed only modified content

Extensible
	•	Future web app can sit on top of:
	•	Same config
	•	Same storage
	•	Same FastAPI server

⸻

Example Use Case

Clone a repository.

Inside it:

rag init
rag index

Now:

rag query "How is JWT validated?"

Or connect an agent to:

rag serve

This turns any project into a structured, searchable knowledge base.

⸻

Roadmap

Milestone 1
	•	Python CLI
	•	Config system
	•	Local indexing
	•	SQLite vector store

Milestone 2
	•	Postgres + pgvector support
	•	Incremental indexing
	•	Retrieval CLI

Milestone 3
	•	FastAPI server mode
	•	Structured retrieval output
	•	Agent integration patterns

Milestone 4
	•	Plugin system
	•	Model switching + re-embedding
	•	Optional web interface

⸻

Positioning

This is a developer tool.

Not a SaaS product.
Not an observability platform.
Not a dashboard.

It exists to answer one question:

“Initialize RAG for this folder.”


