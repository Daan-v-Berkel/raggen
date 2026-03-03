from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import NoResultFound

from .init_config import RagInitConfig
from .engine import create_engine_from_url
from .metadata_schema import metadata, rag_project
from .exceptions import SchemaMismatchError
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
    if diffs:
        raise SchemaMismatchError(f"Stored project configuration differs: {json.dumps(diffs, indent=2)}\nRun with destructive=True to reinitialize.")


def init_database(cfg: RagInitConfig, *, destructive: bool = False) -> Engine:
    engine = create_engine_from_url(cfg.database_url)
    if destructive:
        metadata.drop_all(engine)
    metadata.create_all(engine)

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
            raise SchemaMismatchError(f"Stored project configuration differs: {json.dumps(diffs, indent=2)}\nRun with destructive=True to reinitialize.")
    else:
        # insert new row
        row = cfg.to_row()
        ins = rag_project.insert().values(**row)
        conn.execute(ins)
        conn.commit()
    conn.close()
    return engine
