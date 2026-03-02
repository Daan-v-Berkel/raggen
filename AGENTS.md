# AGENTS.md — Guide for autonomous agents working on this repository

This document records only what was observed in the repository and what an agent needs to know to work effectively here.

---

## Quick check

- Repository root contains: `cli/`, `core/`, `server/`, `docs/`.
- `core/` contains the main implementation (chunking, parsing, logger).
- `cli/` and `server/` are present but contain no files (empty directories in this snapshot).
- No obvious Python packaging or dependency files were found (no `pyproject.toml`, `requirements.txt`, `setup.cfg`, or `Makefile`).
- No tests or CI configuration were found.

If this repo is unexpectedly empty of source code, stop here and add the source first.

---

## High-level purpose (observed)

- The docs describe a "RAG Bootstrapper" (docs/info.md) that provides:
  - a Python RAG engine (parsing, chunking, embedding, indexing)
  - a Python CLI (commands shown in docs)
  - an optional FastAPI server mode

Relevant docs: `docs/info.md` (contains CLI command examples such as `rag init`, `rag index`, `rag query`, `rag serve`).

---

## Observed CLI commands (documented, not implemented in repo)

These commands are described in docs/info.md but no CLI implementation was found in this snapshot:

- `rag init` — interactive setup (create .rag/config etc.)
- `rag index` — run the index pipeline (parse -> chunk -> embed -> store)
- `rag query "..."` — run queries against the index
- `rag serve` — start a lightweight FastAPI server (optional)

Note: The docs explicitly list these commands, but the `cli/` directory is empty here. Do not assume these commands exist until a CLI implementation or entrypoints are found.

---

## Project layout (observed)

- core/
  - logger.py                  — logging setup; writes to `logs/rag-engine.log` (core/logger.py:5-7, get_logger:10-24)
  - chunking/
    - chunks.py                — ChunkConfig, Chunk models and DEFAULT_CHUNK_CONFIG (core/chunking/chunks.py:9-95)
    - chunker.py               — Chunker implementation, validation, offset mapping, chunk enrichment (core/chunking/chunker.py:57-206)
  - parsing/
    - parser.py                — Parser registry, ParserService, helpers for canonical text building (core/parsing/parser.py:37-62, 70-109)
    - PlainTextParser.py       — Plain-text fallback parser implementation (core/parsing/PlainTextParser.py:6-53)
- cli/                         — present but empty in this snapshot
- server/                      — present but empty in this snapshot
- docs/info.md                 — high-level project docs and CLI examples

---

## Key files & useful code locations

- Logging setup: `core/logger.py:5-24` — creates `logs/` and `logs/rag-engine.log`, sets RotatingFileHandler and StreamHandler.
- Chunk models & defaults: `core/chunking/chunks.py:9-95` — Pydantic models, DEFAULT_CHUNK_CONFIG values.
- Chunker core logic: `core/chunking/chunker.py`:
  - STRATEGIES mapping and constructor: `chunker.py:59-67`
  - validate_config with semantic checks: `chunker.py:68-117` (overlap < chunk_size, tokenizer requirement for tokens unit, etc.)
  - stable config hashing: `chunker.py:119-122`
  - deterministic offset finding: `chunker.py:124-141` (_find_offsets_sequential)
  - chunk enrichment + metadata: `chunker.py:164-201`
  - Note: several chunk strategies are unimplemented stubs: `_chunk_heading`, `_chunk_paragraph`, `_chunk_token` (chunker.py:216-223)
- Parsing helpers and registry: `core/parsing/parser.py:37-62`, paragraph splitting and canonical text building: `parser.py:70-109`
- Plain text parser: `core/parsing/PlainTextParser.py:15-53` — decodes bytes using UTF-8 with replacement, normalizes line endings, splits paragraphs using regex (two or more newlines), builds Document with single page spanning whole text.

---

## Observed dependencies (from imports)

- pydantic (used for models throughout core)
- langchain_text_splitters (RecursiveCharacterTextSplitter used in chunker)
- standard library: logging, hashlib, json, re, dataclasses, typing, pathlib

No dependency manifest files were found; an agent should search for a requirements file or pyproject and if none exist, document that dependencies must be installed manually (e.g., `pydantic`, `langchain-text-splitters`, etc.). Do not assume exact package names beyond the import names observed.

---

## Code patterns and conventions (observed)

- Pydantic BaseModel is used for structured configuration/data (ChunkConfig, Chunk, Document, etc.).
- Typing and dataclasses are used for small helper structures (e.g., _BuiltText dataclass in parser.py:64-68).
- Naming: classes use CamelCase, functions and methods use snake_case.
- Canonicalization: paragraph boundaries are defined as \n\n (two-or-more newlines) and paragraphs are rejoined with `"\n\n"` to compute offsets deterministically (`core/parsing/parser.py:74-109`).
- Chunking: a deterministic offset mapping is used to locate chunk spans in the original text (`core/chunking/chunker.py:124-141`).
- Configuration validation: Chunker.validate_config performs both Pydantic validation and semantic checks (e.g., overlap < chunk_size) (core/chunking/chunker.py:68-117).

---

## Testing, build and CI (observed/missing)

- No tests directory or test files were observed.
- No build/packaging files (`pyproject.toml`, `setup.cfg`, `requirements.txt`, `Makefile`) were found.
- No CI configs (.github/workflows) were found.

Implication: there is no observable automated test/run workflow. Agents should not assume a test command exists — create or request tests or packaging if needed.

---

## Important gotchas and non-obvious behaviors

- CLI and server components described in docs are not present in the source tree (directories exist but empty). Verify presence before invoking `rag` commands.
- PlainText parser treats `application/octet-stream` as fallback and returns `effective_mimetype` of `text/plain` for such inputs (`PlainTextParser.py:47-53`).
- Paragraph splitting uses regex r"\n{2,}", which treats two-or-more consecutive newlines as separators; empty paragraphs are dropped (`parser.py:74-85`).
- ChunkConfig defaults set `unit='chars'` and tokenizer name `"none"` (chunks.py:71-95). If switching to token-based chunking, a tokenizer config is required (Chunker.validate_config enforces this, chunker.py:95-99).
- Several chunking strategies are unimplemented (headingAware, paragraphMerge, tokenAware) — callers expecting those may fail at runtime (chunker.py:216-223).
- Chunk offset mapping is sequential and will raise an error if a chunk's text cannot be found in the original document text (chunker.py:124-141).
- Logger creates a `logs/` directory at runtime relative to the package root and writes a rotating log file `logs/rag-engine.log` (logger.py:5-7). Ensure runtime user has filesystem permissions.

---

## Suggested discovery steps for an agent arriving fresh

1. ls -la to confirm files present and check `cli/` and `server/` contents.
2. Search for packaging/dependency files: `glob **/pyproject.toml, **/requirements*.txt, **/setup.cfg`.
3. Search for entrypoints or CLI implementations (look for `if __name__ == "__main__"`, `argparse`, `click`, or `typer` usage) across the repo.
4. Inspect `core/` sources (already useful): chunking and parsing are the core engine components.
5. If implementing CLI or server, follow patterns from `core/` (use Pydantic models, deterministic offsets, respect canonicalization rules in parser).
6. If adding tests or CI, create a minimal `pyproject.toml` or `requirements.txt` listing `pydantic` and `langchain-text-splitters` (use observed imports as the baseline).

---

## Editing & contribution guidelines for agents

- Only edit files after reading them entirely and matching exact whitespace/formatting conventions (this repo uses standard Python indentation; many Pydantic models).
- When changing chunking behavior, update `DEFAULT_CHUNK_CONFIG` (core/chunking/chunks.py:67-95) and ensure `Chunker.validate_config` accepts new combos.
- If you implement missing chunk strategies, add unit tests that verify deterministic offset mapping (`_find_offsets_sequential`) and metadata enrichment behavior.
- When adding new dependencies, add a dependency manifest (`pyproject.toml` or `requirements.txt`) before using them in CI or automation.

---

## Minimal next tasks an agent can perform (observed gaps)

- Implement the CLI (implement commands shown in `docs/info.md`) inside `cli/` and wire entrypoints.
- Add packaging/dependency manifest (pyproject.toml or requirements.txt) listing identified dependencies.
- Add tests covering parsing, canonicalization, and chunker behavior (offset mapping + validation).
- Implement unimplemented chunk strategies in `core/chunking/chunker.py`.

These are suggestions based on repository state; proceed only after confirming with repository owner if unsure.

---

End of AGENTS.md
