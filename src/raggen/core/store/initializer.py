from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.engine import Engine

from raggen.core.config.project import ProjectConfig
from .metadata_schema import rag_project
from .metadata_backends.sqlalchemy import SqlalchemyMetadataBackend
from .exceptions import SchemaMismatchError, BackendLoadError, BackendNotSupportedError
from .plugin_loader import load_vector_backend, resolve_vector_backend_import
from raggen.core.runtime import get_engine
import json


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
        ("database_url", "storage.database_url"),
        ("embedding_model_id", "embedding.model_id"),
        ("embedding_dim", "embedding.dim"),
        ("embedding_normalized", "embedding.normalize"),
        ("query_model_id", "query.model_id"),
    ]

    for stored_k, cfg_path in checks:
        stored_val = getattr(stored, stored_k, "")
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
    with engine.connect() as conn:
        sel = select(rag_project).where(rag_project.c.id == 1)
        try:
            res = conn.execute(sel).mappings().fetchone()
        except Exception:
            # Table may not exist yet; treat as no stored project
            res = None
    if not res:
        return
    diffs = _compare_configs(res, cfg)

    if diffs:
        raise SchemaMismatchError(
            f"Stored project configuration differs: {json.dumps(
                diffs, indent=2)}\nRun with destructive=True to reinitialize."
        )


def _fetch_project_row(engine):
    with engine.connect() as conn:
        try:
            return conn.execute(
                select(rag_project).where(rag_project.c.id == 1)
            ).mappings().fetchone()
        except Exception:
            return None


def _insert_project_row(engine, cfg: ProjectConfig, import_path: str) -> None:
    notes = dict(cfg.notes or {})
    notes["vector_backend_import"] = import_path

    row = cfg.to_row()
    row["notes_json"] = json.dumps(notes)

    with engine.begin() as conn:
        conn.execute(rag_project.insert().values(**row))


def init_database(cfg: ProjectConfig, *, destructive: bool = False) -> Engine:
    engine = get_engine()

    vector_import = resolve_vector_backend_import(
        cfg.storage.backend_key, cfg.storage.vector_backend_import
    )

    try:
        vector_backend = load_vector_backend(vector_import)
    except BackendLoadError:
        raise
    except Exception as exc:
        raise BackendLoadError(
            f"Failed to load vector backend '{vector_import}': {exc}"
        ) from exc

    if not vector_backend.supports(engine):
        raise BackendNotSupportedError(
            f"Backend '{vector_backend.key}' does not support engine dialect '{
                engine.dialect.name}'"
        )

    meta_backend = SqlalchemyMetadataBackend()
    existing = _fetch_project_row(engine)

    if destructive:
        try:
            vector_backend.drop_schema(engine)
        except Exception:
            pass
        meta_backend.drop_schema(engine)
        meta_backend.create_schema(engine)
        vector_backend.create_schema(engine, cfg.embedding.dim)
        _insert_project_row(engine, cfg, vector_import)

    else:
        if existing is not None:
            validate_existing_project(engine, cfg)
        else:
            meta_backend.create_schema(engine)
            vector_backend.create_schema(engine, cfg.embedding.dim)
            _insert_project_row(engine, cfg, vector_import)

    return engine
