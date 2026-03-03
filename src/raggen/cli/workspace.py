import json
import sqlite3
from pathlib import Path
from typing import Optional


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
    if not db_path.exists() or force:
        conn = sqlite3.connect(db_path)
        conn.close()
