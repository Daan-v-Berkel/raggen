from __future__ import annotations

from pathlib import Path
import os
from raggen.core.config.project import default_project_config
from raggen.core.ingest.ingest_service import do_ingest
from raggen.core.store.initializer import init_database
from raggen.core.bootstrap import bootstrap


def run_init(
        *,
        root: str = '.',
        non_interactive: bool = False,
        force: bool = False,
        destructive: bool = False
) -> None:
    root_p = Path(root).resolve()
    cfg = default_project_config(root_p)
    cfg_path = root_p / '.rag' / 'config.toml'

    # Guardrail: refuse to overwrite existing config unless --force is used
    if cfg_path.exists() and not force:
        print(
            "Project already initialised.\nUse --force to overwrite existing configuration.")
        raise SystemExit(1)

    # If force requested and .rag exists, remove it entirely
    if force and (root_p / '.rag').exists():
        print("Overwriting existing project configuration...")
        import shutil

        shutil.rmtree(root_p / '.rag')

    # ensure .rag exists
    os.makedirs(root_p / '.rag', exist_ok=False)

    if non_interactive:
        cfg.save(cfg_path)
        print(f"Wrote config to {cfg_path}")
        return

    # interactive: simple prompt flow
    print(f"Initializing project at {root_p}")
    use_ignorefiles = input(
        "ignorefiles to use (comma separated) [.gitignore]: ").strip().lower() or '.gitignore'
    if use_ignorefiles != '':  # TODO:more robust checking and make optional
        ignorefiles = use_ignorefiles.split(',')
        cfg.scan.ignore_files = ignorefiles
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

    cfg.save(cfg_path)

    print(f"Wrote config to {cfg_path}")
    bootstrap(cfg_path)
    init_database(cfg, destructive=destructive)
    print("initialised database and schema")

    confirm = input(
        "ingest now? (y/n) [n]: ").strip().lower() or 'n'
    if confirm.startswith('y'):
        stats = do_ingest(destructive=False)
        print("Ingest completed:", stats)
    else:
        print("Done. Run `rag ingest` to perform ingestion later.")
