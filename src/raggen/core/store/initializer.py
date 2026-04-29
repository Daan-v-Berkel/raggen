from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.engine import Engine

from raggen.core.config.project import ProjectConfig
from raggen.core.config.drift_tiers import DriftTier, field_tier_info
from .metadata_schema import rag_project
from .metadata_backends.sqlalchemy import SqlalchemyMetadataBackend
from .exceptions import FieldChange, SchemaMismatchError, BackendLoadError, BackendNotSupportedError
from .plugin_loader import load_vector_backend, resolve_vector_backend_import
from raggen.core.runtime import get_engine
import json


def _get_attr_path(obj, path: str):
    value = obj
    for part in path.split("."):
        value = getattr(value, part)
    return value


# Mapping of (stored_column_name, config_field_path) pairs to compare.
# config_field_path is passed to classify_field / field_tier_info to determine
# whether a change is BREAKING, STALE, or RUNTIME.
_COMPARE_FIELDS: list[tuple[str, str]] = [
    ("schema_version",    "schema_version"),
    ("backend_key",       "storage.backend_key"),
    ("database_url",      "storage.database_url"),
    ("embedding_model_id","embedding.model_id"),
    ("embedding_dim",     "embedding.dim"),
    ("embedding_normalized", "embedding.normalize"),
    ("query_model_id",    "query.model_id"),
]


def _collect_changes(stored, cfg: ProjectConfig) -> list[FieldChange]:
    """Compare every tracked field and return a FieldChange for each that differs."""
    changes: list[FieldChange] = []
    for stored_key, field_path in _COMPARE_FIELDS:
        stored_val = getattr(stored, stored_key, None)
        cfg_val = _get_attr_path(cfg, field_path)
        if stored_val != cfg_val:
            info = field_tier_info(field_path)
            changes.append(FieldChange(
                field=field_path,
                old_value=stored_val,
                new_value=cfg_val,
                tier=info.tier,
                reason=info.reason,
            ))
    return changes


def validate_existing_project(engine: Engine, cfg: ProjectConfig) -> None:
    with engine.connect() as conn:
        sel = select(rag_project).where(rag_project.c.id == 1)
        try:
            res = conn.execute(sel).mappings().fetchone()
        except Exception:
            # Table may not exist yet; treat as no stored project.
            res = None
    if not res:
        return

    changes = _collect_changes(res, cfg)

    # Only BREAKING changes require a rebuild.  STALE changes are handled at
    # ingest time (Step 6); RUNTIME changes are silently ignored.
    breaking = [c for c in changes if c.tier == DriftTier.BREAKING]
    if breaking:
        raise SchemaMismatchError(breaking)


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
