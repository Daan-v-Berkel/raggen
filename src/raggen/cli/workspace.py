import json
import sqlite3
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
import hashlib


DEFAULT_CONFIG = {
    "vector_backend": "sqlite_vec",
    "chunking": {
        "strategy": "fixed",
        "chunk_size": 500,
        "overlap": 50
    },
    "embedding_model": "bge-small-en"
}


def _merge_configs(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            merged = dict(out.get(k))
            merged.update(v)
            out[k] = merged
        else:
            out[k] = v
    return out


def _chunk_config_hash(chunk_conf: dict) -> str:
    j = json.dumps(chunk_conf, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(j.encode("utf-8")).hexdigest()


def _init_db(conn: sqlite3.Connection, config: dict) -> None:
    cur = conn.cursor()
    # enforce foreign keys
    cur.execute("PRAGMA foreign_keys = ON;")

    # 1) Project config
    cur.execute('''
    CREATE TABLE IF NOT EXISTS rag_project (
      id INTEGER PRIMARY KEY CHECK (id = 1),
      created_at TEXT NOT NULL,
      schema_version TEXT NOT NULL,
      backend TEXT NOT NULL,
      embedding_model_id TEXT NOT NULL,
      embedding_dim INTEGER NOT NULL,
      embedding_normalized INTEGER NOT NULL,
      query_model_id TEXT,
      chunk_config_hash TEXT NOT NULL,
      notes_json TEXT
    );
    ''')

    # 2) Documents
    cur.execute('''
    CREATE TABLE IF NOT EXISTS documents (
      doc_id TEXT PRIMARY KEY,
      source_path TEXT NOT NULL,
      mimetype TEXT NOT NULL,
      byte_size INTEGER NOT NULL,
      content_hash TEXT NOT NULL,
      parsed_at TEXT NOT NULL,
      parser_id TEXT NOT NULL,
      structure_version TEXT NOT NULL,
      text_char_len INTEGER NOT NULL
    );
    ''')

    # 3) Chunks
    cur.execute('''
    CREATE TABLE IF NOT EXISTS chunks (
      chunk_id TEXT PRIMARY KEY,
      doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
      chunk_index INTEGER NOT NULL,
      text TEXT NOT NULL,
      start_offset INTEGER NOT NULL,
      end_offset INTEGER NOT NULL,
      page_number INTEGER,
      heading_path_json TEXT,
      chunk_config_hash TEXT NOT NULL,
      created_at TEXT NOT NULL
    );
    ''')
    cur.execute(
        'CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks(doc_id);')
    cur.execute(
        'CREATE INDEX IF NOT EXISTS chunks_doc_order_idx ON chunks(doc_id, chunk_index);')

    # 4) Embedding metadata
    cur.execute('''
    CREATE TABLE IF NOT EXISTS embeddings (
      chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
      embedding_model_id TEXT NOT NULL,
      dim INTEGER NOT NULL,
      normalized INTEGER NOT NULL,
      created_at TEXT NOT NULL
    );
    ''')

    # 5) Vectors (sqlite-vec) -- try to create virtual table, fallback to simple blob table
    embedding_dim = 0
    try:
        embedding_model = config.get("embedding_model")
        # We don't know dim at init time; set 0 placeholder. Drivers may accept variable-length.
        embedding_dim = int(config.get("embedding_dim", 0))
    except Exception:
        embedding_dim = 0

    try:
        # Attempt to create sqlite-vec virtual table; some sqlite-vec backends accept FLOAT[<dim>]
        cur.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0( chunk_id TEXT PRIMARY KEY, embedding FLOAT[{embedding_dim}] );")
    except sqlite3.OperationalError:
        # Fallback: store vectors as blob
        cur.execute(
            'CREATE TABLE IF NOT EXISTS chunk_vectors (chunk_id TEXT PRIMARY KEY, embedding BLOB);')

    # Insert or replace project row (single row id=1)
    now = datetime.now(timezone.utc)
    schema_version = "v1"
    backend = config.get("vector_backend", "sqlite_vec")
    embedding_model_id = config.get("embedding_model", "")
    embedding_dim_val = int(config.get("embedding_dim", 0) or 0)
    embedding_normalized = 1  # assume normalized vectors by default
    query_model_id = config.get("query_model")
    chunk_conf = config.get("chunking", {})
    chunk_hash = _chunk_config_hash(chunk_conf)
    notes = json.dumps(config)

    cur.execute('''
    INSERT OR REPLACE INTO rag_project (
      id, created_at, schema_version, backend, embedding_model_id, embedding_dim,
      embedding_normalized, query_model_id, chunk_config_hash, notes_json
    ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    ''', (now, schema_version, backend, embedding_model_id, embedding_dim_val, embedding_normalized, query_model_id, chunk_hash, notes))

    conn.commit()


def init_workspace(root: Path, force: bool = False, config: Optional[dict] = None) -> None:
    rag_dir = root / ".rag"

    if rag_dir.exists() and not force:
        raise RuntimeError(
            ".rag workspace already exists. Use --force to reinitialize.")

    rag_dir.mkdir(exist_ok=True)
    (rag_dir / "indexes").mkdir(exist_ok=True)
    (rag_dir / "logs").mkdir(exist_ok=True)

    # Create VERSION
    (rag_dir / "VERSION").write_text("1\n")

    # Prepare config
    config_path = rag_dir / "config.json"
    config_to_write = dict(DEFAULT_CONFIG)
    if config:
        config_to_write = _merge_configs(config_to_write, config)

    # Write config.json
    if not config_path.exists() or force:
        config_path.write_text(json.dumps(config_to_write, indent=2))

    # Create SQLite DB
    db_path = rag_dir / "rag.db"
    conn = sqlite3.connect(db_path)

    # Initialize schema and vector table
    _init_db(conn, config_to_write)

    conn.close()
