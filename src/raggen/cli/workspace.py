import json
import sqlite3
from pathlib import Path


DEFAULT_CONFIG = {
    "vector_backend": "sqlite_vec",
    "chunking": {
        "strategy": "fixed",
        "chunk_size": 500,
        "overlap": 50
    },
    "embedding_model": "bge-small-en"
}


def init_workspace(root: Path, force: bool = False) -> None:
    rag_dir = root / ".rag"

    if rag_dir.exists() and not force:
        raise RuntimeError(
            ".rag workspace already exists. Use --force to reinitialize.")

    rag_dir.mkdir(exist_ok=True)
    (rag_dir / "indexes").mkdir(exist_ok=True)
    (rag_dir / "logs").mkdir(exist_ok=True)

    # Create VERSION
    (rag_dir / "VERSION").write_text("1\n")

    # Create config.json if missing
    config_path = rag_dir / "config.json"
    if not config_path.exists() or force:
        config_path.write_text(json.dumps(DEFAULT_CONFIG, indent=2))

    # Create SQLite DB
    db_path = rag_dir / "rag.db"
    if not db_path.exists() or force:
        conn = sqlite3.connect(db_path)
        conn.close()
