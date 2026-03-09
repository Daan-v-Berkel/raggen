# AGENTS.md

This file documents what an automated agent (or developer) needs to know to work effectively in this repository.

Status
- Python project packaged with pyproject.toml (setuptools).
- Small RAG pipeline library under src/raggen.
- Tests exist under tests/ and use pytest.

Quick commands
- Activate venv (.venv/bin/activate)
- Install editable for local dev: pip install -e .
- Run tests: pytest -q
- Run CLI help (after install): rag -h

Project entry points
- Console script: rag -> raggen.cli.main:main (defined in pyproject.toml)
- Example usage: src/raggen/cli/main.py defines subcommands: init, ingest, query

Configuration
- Default configuration file path: .rag/config.toml
- Bootstrap uses DEFAULT_CONFIG_PATH = Path('.rag/config.toml') unless explicit path passed.
- Project config model: src/raggen/core/config/project.py (ProjectConfig dataclass)
  - default_project_config(root) builds defaults for a project root
  - to_row() returns a DB-friendly representation used by initializer

Code layout (representative)
- src/raggen/
  - core/
    - bootstrap.py        (project bootstrap, resolves config, creates engine)
    - runtime.py          (global engine holder: set_engine/get_engine)
    - store/              (database schema, backends, initializer)
    - parsing/            (parser base, plaintext fallback)
    - chunking/           (chunker, chunks, pipeline)
    - ingest/             (ingest pipeline, gating, planner)
    - embeddings/         (embedding adapters; not exhaustive)
  - cli/                 (CLI wiring and subcommands)
- scripts/
  - new.py               (ad-hoc e2e scan -> parse -> chunk -> embed script)
  - reinstall.sh         (simple reinstall script)
  - verify_env.sh        (create venv, install, run tests, run rag -h)
- tests/                 (pytest tests for CLI, ingest, store, vector backends)

Notable patterns and conventions
- Packaging: setuptools in pyproject, package-dir maps src/ as root.
- Python version: requires-python = ">=3.12" in pyproject.toml.
- Dataclasses + fancy_dataclass TOML/Config helpers: src/raggen/core/config/project.py
- CLI: argparse-based, main function returns value for query subcommand (see cli/main.py)
- Runtime engine: a module-level singleton in core/runtime.py. Call bootstrap() early to set engine.
- Engine creation: core/store/engine.create_engine_from_url creates SQLAlchemy Engine and wraps its .connect() to accept raw SQL strings (compat layer for SQLAlchemy 2.x).
- Database initializer: core/store/initializer.init_database handles vector backend selection and metadata schema creation. It will:
  - choose vector_backend_import from config or infer from backend_key (sqlite_vec/pgvector)
  - load backend via plugin_loader.load_vector_backend
  - validate backend.supports(engine) and create/drop schemas
  - insert project row (notes_json includes vector_backend_import)
- Guardrails in init command: run_init refuses to overwrite existing .rag/config.toml unless --force is passed. When --force it will remove the .rag dir and recreate.

Testing approach
- Tests use pytest (dev optional-deps lists pytest in pyproject).
- Typical flow in verify_env.sh: create venv, pip install -r requirements.txt, pip install -e ., pytest -q, then rag -h.
- Tests reference many core modules directly (e.g., run_init, init_database), so unit tests expect importable package.

Scripts / helper utilities
- scripts/new.py: standalone script to scan a project tree, parse files using PlainTextFallbackParser, chunk (DEFAULT_CHUNK_CONFIG), and optionally embed chunks using sentence-transformers locally.
  - Embedding requires sentence-transformers to be installed; the script raises if missing.
  - The script implements a small NpyDirCache to cache per-chunk .npy embeddings.
  - Command-line flags: --embed, --embed-model, --embed-batch, --no-normalize, --embed-cache-dir

Dependencies observed (from pyproject.toml)
- sqlite-vec
- numpy==2.4.2
- pydantic==2.12.5
- sentence-transformers==5.2.3
- SQLAlchemy==2.0.48
- fancy-dataclass==0.10.2
- langchain-text-splitters==1.1.1
- tests dev dependency: pytest

Gotchas / Important details
- SQLAlchemy string SQL compatibility: engine.connect() is wrapped so Connection.execute accepts string SQL; be aware when changing engine.connect behavior.
- bootstrap() will raise if .rag/config.toml is missing. Use rag init to initialize a project before calling functions that bootstrap.
- Runtime engine: get_engine() raises RuntimeError if set_engine wasn't called. Many callers assume bootstrap() set the engine.
- init_database has logic to detect existing project rows and will raise SchemaMismatchError if stored DB values differ from config unless destructive=True.
- Some code stores boolean flags as integers in DB (initializer normalizes booleans to 0/1 when comparing).
- Default vector backend and import path are inferred: sqlite_vec -> raggen.core.store.vector_backends.sqlite_vec:SQLiteVecBackend
- tests/ assume local importable package; running tests requires installing the package in editable mode or setting PYTHONPATH to repo root.

Files checked while composing this document
- pyproject.toml
- src/raggen/cli/main.py
- src/raggen/cli/commands/init.py
- src/raggen/core/bootstrap.py
- src/raggen/core/runtime.py
- src/raggen/core/config/project.py
- src/raggen/core/parser/parser.py
- src/raggen/core/parsing/PlainTextParser.py
- src/raggen/core/store/initializer.py
- src/raggen/core/store/engine.py
- scripts/new.py
- scripts/reinstall.sh
- scripts/verify_env.sh
- tests/* (representative)

Missing / not-found
- No CI configs were found in the repository root (e.g., .github/workflows) during the scan.

How to approach common tasks for agents
- Running tests: install package editable then run pytest. scripts/verify_env.sh shows a reproducible sequence.
- Changing DB schema or engine behavior: update core/store/engine.py and core/store/initializer.py together; run tests.
- Adding parsers: implement Parser protocol in src/raggen/core/parsing, register with ParserRegistry, ensure supported_mimetypes set and parse(ParseInput)->ParseResult.
- Working with embeddings: default embedding config is in ProjectConfig.embedding; scripts/new.py shows local embedding flow.

If something looks missing
- Do not invent missing commands; instead search for occurrences (grep) and rely on pyproject and scripts for canonical commands.


-- End of AGENTS.md
