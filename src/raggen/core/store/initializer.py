from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.engine import Engine

from raggen.core.config.project import ProjectConfig
from .metadata_schema import metadata, rag_project
from .exceptions import SchemaMismatchError, BackendLoadError, BackendNotSupportedError
from .plugin_loader import load_vector_backend
from raggen.core.runtime import get_engine
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
        ("storage.backend_key", "storage.backend_key"),
        ("embedding.model_id", "embedding.model_id"),
        ("embedding.dim", "embedding.dim"),
        ("embedding.normalize", "embedding.normalize"),
        ("query.model_id", "query.model_id"),
    ]

    for stored_k, cfg_path in checks:
        stored_val = _get_attr_path(stored, stored_k)
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
    try:
        res = conn.execute(sel).mappings().fetchone()
    except Exception:
        # Table may not exist yet; treat as no stored project
        res = None
    conn.close()
    if not res:
        return
    stored = _row_to_config(res)
    diffs = _compare_configs(stored, cfg)

    if diffs:
        raise SchemaMismatchError(
            f"Stored project configuration differs: {json.dumps(diffs, indent=2)}\nRun with destructive=True to reinitialize."
        )


def init_database(cfg: ProjectConfig, *, destructive: bool = False) -> Engine:
    # Prefer runtime engine when it matches the requested DB URL; otherwise create a
    # transient engine for the requested cfg.storage.database_url so callers that
    # don't bootstrap still work against the intended database.
    # TODO: this is shit, callers need to call bootstrap, this only exists for tests to work. Need a betterway to handle testing
    try:
        engine = get_engine()
        try:
            # compare configured URL to current engine url and create a local
            # engine for this cfg if they differ
            from raggen.core.store.engine import create_engine_from_url

            current_url = getattr(engine, "url", None)
            if current_url is not None and str(current_url) != cfg.storage.database_url:
                engine = create_engine_from_url(cfg.storage.database_url)
        except Exception:
            # if anything goes wrong comparing/creating engine, fall back to the
            # runtime engine
            pass
    except RuntimeError:
        # no runtime engine registered; create one from cfg
        from raggen.core.store.engine import create_engine_from_url

        engine = create_engine_from_url(cfg.storage.database_url)

    # determine import path: if not provided, pick built-in by backend_key
    import_path = cfg.storage.vector_backend_import
    if not import_path:
        if cfg.storage.backend_key == "sqlite_vec":
            import_path = (
                "raggen.core.store.vector_backends.sqlite_vec:SQLiteVecBackend"
            )
        elif cfg.storage.backend_key == "pgvector":
            import_path = "raggen.core.store.vector_backends.pgvector:PgVectorBackend"
        else:
            raise BackendLoadError(
                f"No vector_backend_import provided and unknown backend_key '{cfg.storage.backend_key}'"
            )

    # load vector backend
    try:
        backend = load_vector_backend(import_path)
    except Exception as exc:
        raise BackendLoadError(
            f"Failed to load vector backend '{import_path}': {exc}"
        ) from exc

    # ensure backend supports this engine
    if not backend.supports(engine):
        raise BackendNotSupportedError(
            f"Backend '{backend.key}' does not support engine dialect '{engine.dialect.name}'"
        )

    # Check existing project row BEFORE creating vector schema to avoid backend-side limits
    # conn = engine.connect()
    # sel = select(rag_project).where(rag_project.c.id == 1)
    # try:
    #     res = conn.execute(sel).mappings().fetchone()
    # except Exception:
    #     # Table may not exist yet; treat as no stored project
    #     res = None
    if not destructive:
        validate_existing_project(engine, cfg)

    else:
        # If destructive, drop vector schema first (safer) then drop metadata
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
