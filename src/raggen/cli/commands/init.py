from __future__ import annotations

from pathlib import Path
import os
from ...core.ingest.config import ProjectConfig, save_project_config, default_project_config
from ...core.ingest.ingest_service import init_and_ingest


def run_init(*, root: str = '.', non_interactive: bool = False, destructive: bool = False) -> None:
    root_p = Path(root).resolve()
    cfg = default_project_config(root_p)
    cfg_path = root_p / '.rag' / 'config.toml'
    os.makedirs(root_p / '.rag', exist_ok=True)
    if non_interactive:
        save_project_config(cfg, cfg_path)
        print(f"Wrote config to {cfg_path}")
        return
    # interactive: simple prompt flow
    print(f"Initializing project at {root_p}")
    use_git = input("Use .gitignore? (y/n) [y]: ").strip().lower() or 'y'
    if use_git.startswith('y'):
        cfg.scan.use_gitignore = True
    extra = input("Additional ignore globs (comma-separated) []: ").strip()
    if extra:
        cfg.scan.ignore = [s.strip() for s in extra.split(',') if s.strip()]
    # chunking
    ch_size = input(f"Chunk size [{cfg.chunking.chunk_size}]: ").strip()
    if ch_size:
        cfg.chunking.chunk_size = int(ch_size)
    ch_overlap = input(f"Chunk overlap [{cfg.chunking.overlap}]: ").strip()
    if ch_overlap:
        cfg.chunking.overlap = int(ch_overlap)
    # embedding
    emb_model = input(
        f"Embedding model id [{cfg.embedding.model_id}]: ").strip()
    if emb_model:
        cfg.embedding.model_id = emb_model
    emb_dim = input(f"Embedding dimension [{cfg.embedding.dim}]: ").strip()
    if emb_dim:
        cfg.embedding.dim = int(emb_dim)
    # storage
    backend = input(
        f"Storage backend key [{cfg.storage.backend_key}]: ").strip()
    if backend:
        cfg.storage.backend_key = backend
    db_url = input(f"Database URL [{cfg.storage.database_url}]: ").strip()
    if db_url:
        cfg.storage.database_url = db_url
    vbi = input(
        f"Vector backend import [{cfg.storage.vector_backend_import}]: ").strip()
    if vbi:
        cfg.storage.vector_backend_import = vbi

    save_project_config(cfg, cfg_path)
    print(f"Wrote config to {cfg_path}")

    confirm = input(
        "Initialize DB and ingest now? (y/n) [n]: ").strip().lower() or 'n'
    if confirm.startswith('y'):
        stats = init_and_ingest(cfg=cfg, destructive_init=destructive)
        print("Ingest completed:", stats)
    else:
        print("Done. Run `rag ingest` to perform ingestion later.")
