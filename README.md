# raggen

Local-first RAG (retrieval-augmented generation) pipeline library and CLI.

Lightweight toolkit for scanning a project tree, parsing files, chunking text, and storing and embedding chunks with pluggable vector backends.
stores used models for further offline usability, or use already local models out of the box.
Use your own vector backend by writing a lightweight plugin, see [### Custom vector backends] for details.

---

## Requirements

- Python 3.12+
- See `pyproject.toml` for runtime dependencies

---

## Quickstart

```bash
# 1. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install (editable for development)
pip install -e .

# 3. Run tests
pytest -q

# 4. See available CLI commands
rag -h
```

---

## Common workflows

```bash
# Initialise a project — creates .rag/config.toml
rag init

# Build the database schema
rag build

# Ingest files from the current directory
rag ingest

# Query the index
rag query "what does the config system do?"
```

---

## Parsing

When a file is ingested, raggen selects a parser based on the file's MIME type
and runs it before any chunking takes place. Every parser produces a
`Document` whose `text` field contains **Markdown-flavoured plain text**:

- Headings are expressed as `#`, `##`, `###` prefixes
- Paragraphs are separated by `\n\n`

### Built-in parsers

| MIME type | File extensions | Behaviour |
|-----------|----------------|-----------|
| `text/markdown` | `.md`, `.markdown` | Pass-through — Markdown headings are already in the correct format |
| `text/plain` | `.txt` and unknown types | Normalises line endings, collapses multiple blank lines to `\n\n`, no heading markers added |
| `text/html`, `application/xhtml+xml` | `.html`, `.htm`, `.xhtml` | Converts `<h1>`–`<h3>` to `#`/`##`/`###`, strips `<script>` and `<style>`, turns `<pre>` into fenced code blocks |

Files with an unrecognised MIME type fall back to the plain text parser.

Future parsers (e.g. DOCX, RST) follow the same convention: convert
format-specific heading markers to `#` prefixes so the `headingAware`
chunking strategy works without modification.

---

## Chunking

Chunking controls how parsed documents are split into pieces before embedding.
raggen uses a **file-group** system: you assign file extensions to named groups
in the config, and each group gets its own chunking strategy.

### File groups

```toml
[file_groups]
# The fallback group catches every extension not listed elsewhere.
# Its name is set by fallback_group below — it does not have to be "fallback".
fallback = { extensions = [] }

docs = { extensions = [".md", ".html", ".txt"] }
code = { extensions = [".py", ".ts", ".js", ".go", ".rs"] }
```

`fallback_group` names which group is used when a file's extension is not
listed in any group:

```toml
fallback_group = "fallback"
```

Every group defined in `[file_groups]` **must** have a matching entry in
`[chunking]`. Missing or extra entries are a configuration error caught on
startup.

### Chunking strategies

Each group's strategy is set independently:

```toml
[chunking.fallback]
strategy  = "fixed"
unit      = "chars"
chunk_size = 1000
overlap    = 100

[chunking.docs]
strategy  = "headingAware"
unit      = "chars"
chunk_size = 1200
overlap    = 150

[chunking.code]
strategy  = "codeAware"
unit      = "tokens"
chunk_size = 256
overlap    = 32
```

#### `fixed`

Splits text into equal-sized chunks using `RecursiveCharacterTextSplitter`.
Tries progressively smaller separators (paragraph, line, word, character)
until each piece fits within `chunk_size`. Good all-purpose default.

#### `paragraphMerge`

Splits exclusively on double newlines (`\n\n`), then merges adjacent
paragraphs until `chunk_size` is reached. Keeps paragraphs intact.
Well suited for prose-heavy plain text.

#### `headingAware`

Splits on Markdown heading markers (`#`, `##`, `###`) first, grouping all
content under each heading into one section. Sections that exceed `chunk_size`
are sub-split using `RecursiveCharacterTextSplitter`. Every sub-chunk carries
the full heading path as metadata (e.g. `["Introduction", "Installation",
"Requirements"]`) so retrieval results can always show which section a chunk
came from.

Works with any file format whose parser emits `#`-prefixed headings — `.md`
natively, `.html` via the HTML parser, and any future parser that follows the
same convention. Plain text files with no headings degrade gracefully to
paragraph-level splitting.

#### `codeAware`

Uses `RecursiveCharacterTextSplitter.from_language()` with language-specific
separators. For Python this means split points are tried in order:
`\nclass `, `\ndef `, `\n\tdef `, blank lines, then lines and characters.
Functions and classes are kept intact wherever the chunk size allows.

Supported languages are detected from the file extension: Python, JavaScript,
TypeScript, Go, Rust, Ruby, Java, C, C++, C#, Swift, Kotlin, Scala, PHP, R,
Lua, Perl, Elixir, Haskell, Solidity, Protobuf, PowerShell. Files with an
unrecognised extension fall back to fixed splitting.

### Sizing unit — `chars` vs `tokens`

`chunk_size` and `overlap` can be measured in characters or in tokens:

```toml
[chunking.docs]
strategy   = "headingAware"
unit       = "chars"    # default — chunk_size is a character count
chunk_size = 1200
overlap    = 150
```

```toml
[chunking.code]
strategy   = "codeAware"
unit       = "tokens"   # chunk_size is a token count
chunk_size = 256        # fits within most 512-token embedding models
overlap    = 32
```

When `unit = "tokens"`, raggen uses the tokenizer that is **bundled with the
configured embedding model** — no extra downloads or dependencies. Both
`chunk_size` and `overlap` are measured using the same tokenizer, including
inside the `headingAware` section-size check and sub-splitter.

The token count excludes special tokens (CLS, SEP) so `chunk_size` maps
directly to content tokens. Account for special tokens when setting a limit
close to your model's maximum sequence length — for a 512-token model a safe
`chunk_size` is around `500`.

---

## Vector backends

raggen ships with two vector backends out of the box:

| Key | Description |
|-----|-------------|
| `sqlite_vec` | Local SQLite database using the sqlite-vec extension. Good default for single-machine use. No extra infrastructure needed. |
| `pgvector` | PostgreSQL with the pgvector extension. Good for shared or production deployments. |

`sqlite_vec` works out of the box. For pgvector, install the PostgreSQL driver:

```bash
pip install raggen[postgres]
```

Configure the backend in `.rag/config.toml`:

```toml
# SQLite (default)
[storage]
backend_key  = "sqlite_vec"
database_url = "sqlite:///.rag/rag.db"
```

```toml
# PostgreSQL
[storage]
backend_key  = "pgvector"
database_url = "postgresql://user:pass@localhost/mydb"
```

### Custom vector backends

Bring your own vector store (Qdrant, Chroma, Weaviate, a proprietary system,
etc.) by implementing the `VectorBackend` interface and pointing the config at
your class:

```toml
[storage]
backend_key           = "my_backend"
vector_backend_import = "my_package.backends:MyVectorBackend"
```

raggen imports and instantiates your class at runtime. The interface requires
six methods: `supports`, `create_schema`, `drop_schema`, `upsert_vectors`,
`delete_vectors`, and `search`.

The full contract — including the transaction model, score conventions, and
implementation notes — is documented in:

```
src/raggen/core/store/vector_backends/base.py
```

> **Important:** `database_url` is not passed to your backend. raggen uses it
> only to connect to the metadata database. Your backend is responsible for
> its own connection — read a URL from an environment variable, a separate
> config file, or hardcode it for local development. The `Engine` and
> `Connection` objects your methods receive belong to the metadata database
> and must not be used to talk to your vector store.

---

## Metadata storage

Document metadata (file paths, chunk text, embedding records) is always stored
via SQLAlchemy using the `database_url` from `[storage]`. Any database
supported by SQLAlchemy works here without additional code.

For the built-in backends (`sqlite_vec`, `pgvector`) vector data lives in the
same database as the metadata, so `database_url` covers both. For a custom
backend that connects to a separate service, `database_url` covers metadata
only.

---

## License

MIT — see `LICENSE`
