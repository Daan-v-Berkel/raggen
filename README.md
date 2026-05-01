# raggen

Index and search your files locally. No cloud, no API keys — just your files, embedded on your machine.

```bash
pip install raggen

rag init .
rag build
rag ingest
rag query "how does the config system work?"
```

---

## Contents

- [Installation](#installation)
- [Commands](#commands)
- [Configuration](#configuration)
- [Supported file types](#supported-file-types)
- [Chunking strategies](#chunking-strategies)
- [Embedding model](#embedding-model)
- [Vector backends](#vector-backends)
- [Metadata storage](#metadata-storage)

---

## Installation

**Requires Python 3.12+**

```bash
pip install raggen
```

The default install uses [fastembed](https://github.com/qdrant/fastembed) (ONNX Runtime) for embeddings — no GPU drivers or CUDA stack required. The first `rag build` downloads a small ONNX model (~23 MB).

**Optional extras:**

| Extra | What it adds | When to use |
|---|---|---|
| `pip install 'raggen[torch]'` | sentence-transformers + PyTorch | Models not in fastembed's registry, or explicit PyTorch preference |
| `pip install 'raggen[postgres]'` | PostgreSQL driver | PostgreSQL vector backend |

After installing `raggen[torch]`, activate it in `.rag/config.toml`:

```toml
[embedding]
backend = "torch"
```

> fastembed is always installed as a core dependency. With `backend = "torch"` it is never imported at runtime — it is unused disk space only (~43 MB).

---

## Commands

raggen is a four-step pipeline. Run the steps in order the first time; after that `rag ingest` and `rag query` are the day-to-day commands.

### `rag init <directory>`

Creates a `.rag/` folder in the given directory containing a `config.toml` with sensible defaults. Edit this file to change the embedding model, storage backend, or chunking settings.

```bash
rag init .              # initialise the current directory
rag init ./myproject
```

### `rag build`

Sets up the database schema. Run once after `rag init`. If you later change a foundational setting (embedding model, storage backend), run it again.

```bash
rag build
rag build --destructive   # wipe and rebuild — needed after breaking config changes
```

### `rag ingest`

Scans your files, parses and chunks them, generates embeddings, and stores everything locally. Only files that have changed since the last run are processed.

```bash
rag ingest
rag ingest --force        # re-index everything, useful after changing chunking config
```

### `rag query`

Runs a semantic similarity search and returns the most relevant chunks.

```bash
rag query "what does the authentication module do?"
rag query "database schema" --top-k 5
```

### `rag runs`

Inspect the history of past commands.

```bash
rag runs list                              # list recent runs
rag runs show --latest                     # show the last run in detail
rag runs show --latest --operation ingest  # show the last ingest run
rag runs show <run-id>                     # show a specific run by ID
```

### `rag --version`

Print the installed version.

```bash
rag --version
```

---

## Configuration

`rag init` writes `.rag/config.toml` with defaults. The key sections:

```toml
[embedding]
model_id = "sentence-transformers/all-MiniLM-L6-v2"
backend  = "auto"   # "auto" | "onnx" | "torch"
normalize = true

[storage]
backend_key  = "sqlite_vec"
database_url = "sqlite:///.rag/rag.db"

[scan]
# Glob patterns to exclude from indexing
ignore = ["node_modules/**", ".git/**", "*.lock"]

[file_groups]
# Assign file extensions to named groups.
# The fallback group catches anything not listed elsewhere.
fallback = { extensions = [] }
docs     = { extensions = [".md", ".txt", ".html"] }
code     = { extensions = [".py", ".ts", ".js", ".go"] }

fallback_group = "fallback"

[chunking.fallback]
strategy   = "fixed"
unit       = "chars"
chunk_size = 1000
overlap    = 100

[chunking.docs]
strategy   = "headingAware"
unit       = "chars"
chunk_size = 1200
overlap    = 150

[chunking.code]
strategy   = "codeAware"
unit       = "tokens"
chunk_size = 256
overlap    = 32
```

Every group defined in `[file_groups]` must have a corresponding `[chunking.<group>]` section, and vice versa. Missing or extra entries are caught as a configuration error on startup.

---

## Supported file types

raggen selects a parser based on file MIME type. All parsers produce plain text with Markdown-style headings (`#`, `##`, `###`) and paragraphs separated by `\n\n`. This uniform format means chunking strategies like `headingAware` work the same way across all supported file types.

| MIME type | Extensions | Behaviour |
|---|---|---|
| `text/markdown` | `.md`, `.markdown` | Passed through as-is — headings are already in the correct format |
| `text/plain` | `.txt`, unknown types | Line endings normalised, multiple blank lines collapsed to `\n\n` |
| `text/html`, `application/xhtml+xml` | `.html`, `.htm`, `.xhtml` | `<h1>`–`<h3>` converted to `#`/`##`/`###`, `<script>` and `<style>` stripped, `<pre>` blocks converted to fenced code |

Files with an unrecognised MIME type fall back to the plain text parser.

---

## Chunking strategies

Chunking controls how parsed files are split into pieces before embedding. raggen uses a **file-group** system: you assign file extensions to named groups in the config, and each group gets its own strategy and sizing.

### `fixed`

Splits text into equal-sized pieces. Tries progressively smaller separators (paragraph → line → word → character) until each piece fits within `chunk_size`. Good all-purpose default for most content types.

### `paragraphMerge`

Splits on double newlines, then merges adjacent paragraphs up to `chunk_size`. Keeps paragraphs intact. Well suited for prose-heavy plain text.

### `headingAware`

Splits on Markdown headings (`#`, `##`, `###`), grouping all content under each heading into one section. Sections larger than `chunk_size` are further split. Each chunk carries its full heading path as metadata (e.g. `["Setup", "Installation", "Requirements"]`), so results always show which section they came from.

Works with any format whose parser emits `#`-prefixed headings — Markdown natively, HTML via the built-in parser. Plain text files with no headings degrade gracefully to paragraph-level splitting.

### `codeAware`

Uses language-aware split points: class boundaries, function definitions, blank lines, then individual lines. Functions and classes are kept intact wherever the chunk size allows.

Supported languages are detected from the file extension: Python, JavaScript, TypeScript, Go, Rust, Ruby, Java, C, C++, C#, Swift, Kotlin, Scala, PHP, R, Lua, Perl, Elixir, Haskell, Solidity, Protobuf, PowerShell. Files with an unrecognised extension fall back to `fixed`.

### Sizing: `chars` vs `tokens`

`chunk_size` and `overlap` can be measured in characters or in tokens:

```toml
[chunking.docs]
strategy   = "headingAware"
unit       = "chars"      # chunk_size is a character count
chunk_size = 1200
overlap    = 150
```

```toml
[chunking.code]
strategy   = "codeAware"
unit       = "tokens"     # chunk_size is a token count
chunk_size = 256          # fits within most 512-token embedding models
overlap    = 32
```

When `unit = "tokens"`, raggen uses the tokenizer bundled with the configured embedding model — no extra downloads. `chunk_size` and `overlap` are both measured with that tokenizer, including inside `headingAware`'s section-size check.

---

## Embedding model

raggen supports two embedding backends, selectable per project in `.rag/config.toml`:

```toml
[embedding]
model_id = "sentence-transformers/all-MiniLM-L6-v2"
backend  = "auto"   # "auto" | "onnx" | "torch"
```

| Backend | Installed by | Default | Notes |
|---|---|---|---|
| `onnx` | `pip install raggen` | Yes | ONNX Runtime, no PyTorch. Fast, lightweight. |
| `torch` | `pip install 'raggen[torch]'` | No | Full sentence-transformers + PyTorch. Supports any HuggingFace model. |

`backend = "auto"` uses ONNX if fastembed is installed, falls back to torch.

The model is downloaded on first use and cached under `model_cache_dir` (default `.rag/models/`). Subsequent runs load it from cache with no network access.

**Switching backends on an existing project** invalidates stored embeddings — run `rag build --destructive && rag ingest` to rebuild. raggen detects this automatically and raises an error before any data is touched.

### Sequence length limits

Every embedding model has a hard maximum sequence length (commonly 256 or 512 tokens). Chunks that exceed this limit are **silently truncated by the model** — the stored text is complete, but the embedding only represents the first N tokens. This degrades retrieval quality without any visible error.

raggen guards against this in two ways:

- **`unit = "tokens"`** — raggen compares `chunk_size` against the model's limit at startup and raises an error before any file is processed.

- **`unit = "chars"`** — an exact check is not possible upfront because the token count varies by content. raggen applies a conservative estimate of 3 chars per token: if your `chunk_size` implies chunks that could exceed the model limit, a warning is emitted at startup. Reduce `chunk_size` or switch to `unit = "tokens"` to eliminate it.

> **Tip:** Use `unit = "tokens"` with `chunk_size` set a few tokens below the model's maximum for a precise, content-independent guarantee. For a 512-token model, `chunk_size = 500` is a safe starting point.

---

## Vector backends

Two backends are included:

| Key | Description |
|---|---|
| `sqlite_vec` | Local SQLite file using the sqlite-vec extension. Default. No extra infrastructure needed. |
| `pgvector` | PostgreSQL with the pgvector extension. Better suited for shared or production deployments. |

### SQLite (default)

Works out of the box, no additional installation required:

```toml
[storage]
backend_key  = "sqlite_vec"
database_url = "sqlite:///.rag/rag.db"
```

### PostgreSQL

Install the driver first:

```bash
pip install raggen[postgres]
```

```toml
[storage]
backend_key  = "pgvector"
database_url = "postgresql://user:pass@localhost/mydb"
```

### Custom backends

Bring your own vector store (Qdrant, Chroma, Weaviate, etc.) by implementing the `VectorBackend` interface and pointing the config at your class:

```toml
[storage]
backend_key           = "my_backend"
vector_backend_import = "my_package.backends:MyVectorBackend"
```

raggen loads your class at runtime. The interface requires six methods: `supports`, `create_schema`, `drop_schema`, `upsert_vectors`, `delete_vectors`, and `search`. The full contract — including the transaction model, score conventions, and implementation notes — is in `src/raggen/core/store/vector_backends/base.py`.

> **Important:** `database_url` is not passed to your backend. raggen uses it only to connect to the metadata database. Your backend is responsible for its own connection — read a URL from an environment variable, a separate config file, or hardcode it for local development. The `Engine` and `Connection` objects your methods receive belong to the metadata database and must not be used to talk to your vector store.

---

## Metadata storage

Document metadata (file paths, chunk text, embedding records) is always stored via SQLAlchemy using the `database_url` from `[storage]`. Any database supported by SQLAlchemy works here without additional code.

For the built-in backends (`sqlite_vec`, `pgvector`) vector data lives in the same database as the metadata, so `database_url` covers both. For a custom backend that connects to a separate service, `database_url` covers metadata only — your backend manages its own connection.

---

## License

MIT — see `LICENSE`
