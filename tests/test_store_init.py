import json
from pathlib import Path
import pytest
from sqlalchemy import inspect

from raggen.core.store import RagInitConfig, init_database, SchemaMismatchError


def make_cfg(db_path: Path, **kwargs):
    return RagInitConfig(
        database_url=f"sqlite:///{db_path}",
        embedding_model_id=kwargs.get("embedding_model_id", "bge-small-en"),
        embedding_dim=kwargs.get("embedding_dim", 1536),
        embedding_normalized=kwargs.get("embedding_normalized", True),
        chunk_config_hash=kwargs.get("chunk_config_hash", "abc123"),
        notes=kwargs.get("notes", {}),
    )


def test_init_creates_tables_and_project_row(tmp_path):
    db_file = tmp_path / "rag.db"
    cfg = make_cfg(db_file)
    engine = init_database(cfg)
    insp = inspect(engine)
    assert "rag_project" in insp.get_table_names()
    assert "documents" in insp.get_table_names()
    assert "chunks" in insp.get_table_names()
    assert "embeddings" in insp.get_table_names()

    # verify project row
    with engine.connect() as conn:
        row = conn.execute("SELECT * FROM rag_project WHERE id=1").mappings().fetchone()
        assert row is not None
        assert row["embedding_model_id"] == cfg.embedding_model_id
        assert int(row["embedding_dim"]) == cfg.embedding_dim


def test_init_validates_existing_config(tmp_path):
    db_file = tmp_path / "rag.db"
    cfg1 = make_cfg(db_file, embedding_dim=1234)
    init_database(cfg1)
    # same config should be fine
    init_database(cfg1)
    # different config should raise
    cfg2 = make_cfg(db_file, embedding_dim=9999)
    with pytest.raises(SchemaMismatchError):
        init_database(cfg2)


def test_destructive_reinit_allows_new_config(tmp_path):
    db_file = tmp_path / "rag.db"
    cfg1 = make_cfg(db_file, embedding_dim=1234)
    init_database(cfg1)
    cfg2 = make_cfg(db_file, embedding_dim=9999)
    init_database(cfg2, destructive=True)
    # verify now matches cfg2
    engine = init_database(cfg2)
    with engine.connect() as conn:
        row = conn.execute("SELECT embedding_dim FROM rag_project WHERE id=1").mappings().fetchone()
        assert row and int(row["embedding_dim"]) == cfg2.embedding_dim
