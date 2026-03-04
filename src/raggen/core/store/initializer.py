from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.engine import Engine

from .init_config import RagInitConfig
from .engine import create_engine_from_url
from .metadata_schema import metadata, rag_project
from .exceptions import SchemaMismatchError, BackendLoadError, BackendNotSupportedError
from .plugin_loader import load_vector_backend
import json


def _row_to_config(row) -> dict:
    # row is a RowMapping; convert to normal dict
    return dict(row)


def _compare_configs(stored: dict, cfg: RagInitConfig) -> dict:
    diffs = {}
    checks = [
        ("schema_version", "schema_version"),
        ("backend_key", "backend_key"),
        ("embedding_model_id", "embedding_model_id"),
        ("embedding_dim", "embedding_dim"),
        ("embedding_normalized", "embedding_normalized"),
        ("query_model_id", "query_model_id"),
        ("chunk_config_hash", "chunk_config_hash"),
    ]
    for stored_k, cfg_k in checks:
        stored_val = stored.get(stored_k)
        cfg_val = getattr(cfg, cfg_k)
        # normalize numeric/bool
        if isinstance(cfg_val, bool):
            cfg_val = 1 if cfg_val else 0
        if stored_val != cfg_val:
            diffs[stored_k] = {"stored": stored_val, "expected": cfg_val}
    return diffs


def validate_existing_project(engine: Engine, cfg: RagInitConfig) -> None:
    conn = engine.connect()
    sel = select(rag_project).where(rag_project.c.id == 1)
    res = conn.execute(sel).mappings().fetchone()
    conn.close()
    if not res:
        return
    stored = _row_to_config(res)
    diffs = _compare_configs(stored, cfg)
    # also validate stored notes for vector backend import path if present
    stored_notes = {}
    try:
        stored_notes = json.loads(stored.get("notes_json") or "{}")
    except Exception:
        stored_notes = {}
    stored_vbi = stored_notes.get("vector_backend_import")
    if stored_vbi and stored_vbi != cfg.vector_backend_import:
        diffs["vector_backend_import"] = {
            "stored": stored_vbi, "expected": cfg.vector_backend_import}
    if diffs:
        raise SchemaMismatchError(
            f"Stored project configuration differs: {json.dumps(diffs, indent=2)}\nRun with destructive=True to reinitialize.")


def init_database(cfg: RagInitConfig, *, destructive: bool = False) -> Engine:
    engine = create_engine_from_url(cfg.database_url)

    # determine import path: if not provided, pick built-in by backend_key
    import_path = cfg.vector_backend_import
    if not import_path:
        if cfg.backend_key == "sqlite_vec":
            import_path = "raggen.core.store.vector_backends.sqlite_vec:SQLiteVecBackend"
        elif cfg.backend_key == "pgvector":
            import_path = "raggen.core.store.vector_backends.pgvector:PgVectorBackend"
        else:
            raise BackendLoadError(
                f"No vector_backend_import provided and unknown backend_key '{cfg.backend_key}'")

    # load vector backend
    try:
        backend = load_vector_backend(import_path)
    except Exception as exc:
        raise BackendLoadError(
            f"Failed to load vector backend '{import_path}': {exc}") from exc

    # ensure backend supports this engine
    if not backend.supports(engine):
        raise BackendNotSupportedError(
            f"Backend '{backend.key}' does not support engine dialect '{engine.dialect.name}'")

    # If destructive, drop vector schema first (safer) then drop metadata
    if destructive:
        try:
            backend.drop_schema(engine)
        except Exception:
            # ignore errors during drop to allow metadata drop to proceed
            pass
        metadata.drop_all(engine)

    # create metadata tables
    metadata.create_all(engine)

    # create vector schema
    backend.create_schema(engine, cfg.embedding_dim)

    # check existing project row
    conn = engine.connect()
    sel = select(rag_project).where(rag_project.c.id == 1)
    res = conn.execute(sel).mappings().fetchone()

    if res and not destructive:
        # validate
        stored = dict(res)
        diffs = _compare_configs(stored, cfg)
        if diffs:
            conn.close()
            raise SchemaMismatchError(
                f"Stored project configuration differs: {json.dumps(diffs, indent=2)}\nRun with destructive=True to reinitialize.")
    else:
        # insert new row, include vector_backend_import in notes
        notes = dict(cfg.notes or {})
        notes["vector_backend_import"] = cfg.vector_backend_import
        cfg.notes = notes
        row = cfg.to_row()
        ins = rag_project.insert().values(**row)
        conn.execute(ins)
        conn.commit()
    conn.close()
    # attach backend to engine for callers, but return engine for backwards compatibility
    try:
        setattr(engine, "_rag_vector_backend", backend)
    except Exception:
        pass
    return engine
