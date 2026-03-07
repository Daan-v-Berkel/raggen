from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.engine import Engine

from raggen.core.config.project import ProjectConfig
from .metadata_schema import metadata, rag_project
from .exceptions import SchemaMismatchError, BackendLoadError, BackendNotSupportedError
from .plugin_loader import load_vector_backend
from raggen.core.runtime import get_engine, set_engine
import json


def _row_to_config(row) -> ProjectConfig:
    # row is a RowMapping; convert to normal dict
    return ProjectConfig.from_dict(dict(row))


def _get_attr_path(obj, path: str):
    value = obj
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _compare_configs(stored: dict, cfg: ProjectConfig) -> dict:
    diffs = {}
    checks = [
        ("schema_version", "schema_version"),
        ("backend_key", "storage.backend_key"),
        ("embedding_model_id", "embedding.model_id"),
        ("embedding_dim", "embedding.dim"),
        ("embedding_normalized", "embedding.normalize"),
        ("query_model_id", "query.model_id"),
    ]

    for stored_k, cfg_path in checks:
        stored_val = stored.get(stored_k)
        cfg_val = _get_attr_path(cfg, cfg_path)

        # normalize bool to match DB integer storage
        if isinstance(cfg_val, bool):
            cfg_val = 1 if cfg_val else 0

        if stored_val != cfg_val:
            diffs[stored_k] = {
                "stored": stored_val,
                "expected": cfg_val,
            }

    return diffs


def validate_existing_project(engine: Engine, cfg: ProjectConfig) -> None:
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
    if stored_vbi and stored_vbi != cfg.storage.vector_backend_import:
        diffs["vector_backend_import"] = {
            "stored": stored_vbi, "expected": cfg.storage.vector_backend_import}
    if diffs:
        raise SchemaMismatchError(
            f"Stored project configuration differs: {json.dumps(diffs, indent=2)}\nRun with destructive=True to reinitialize.")


def init_database(cfg: ProjectConfig, *, destructive: bool = False) -> Engine:
    engine = get_engine()

    # determine import path: if not provided, pick built-in by backend_key
    import_path = cfg.storage.vector_backend_import
    if not import_path:
        if cfg.storage.backend_key == "sqlite_vec":
            import_path = "raggen.core.store.vector_backends.sqlite_vec:SQLiteVecBackend"
        elif cfg.storage.backend_key == "pgvector":
            import_path = "raggen.core.store.vector_backends.pgvector:PgVectorBackend"
        else:
            raise BackendLoadError(
                f"No vector_backend_import provided and unknown backend_key '{cfg.storage.backend_key}'")

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

    # Check existing project row BEFORE creating vector schema to avoid backend-side limits
    conn = engine.connect()
    sel = select(rag_project).where(rag_project.c.id == 1)
    try:
        res = conn.execute(sel).mappings().fetchone()
    except Exception:
        # Table may not exist yet; treat as no stored project
        res = None

    if res and not destructive:
        # validate
        stored = dict(res)
        diffs = _compare_configs(stored, cfg)
        conn.close()
        if diffs:
            raise SchemaMismatchError(
                f"Stored project configuration differs: {json.dumps(diffs, indent=2)}\nRun with destructive=True to reinitialize.")
    else:
        # If destructive, drop vector schema first (safer) then drop metadata
        conn.close()
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
        backend.create_schema(engine, cfg.embedding.dim)

        # insert new row, include vector_backend_import in notes
        notes = dict(cfg.notes or {})
        notes["vector_backend_import"] = cfg.storage.vector_backend_import
        cfg.notes = notes
        row = cfg.to_row()
        with engine.begin() as conn2:
            ins = rag_project.insert().values(**row)
            conn2.execute(ins)
    # attach backend to engine for callers, but return engine for backwards compatibility
    try:
        setattr(engine, "_rag_vector_backend", backend)
    except Exception:
        pass
    return engine
    # attach backend to engine for callers, but return engine for backwards compatibility
    try:
        setattr(engine, "_rag_vector_backend", backend)
    except Exception:
        pass
    return engine
