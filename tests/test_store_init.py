from pathlib import Path
import pytest
from sqlalchemy import inspect

from raggen.core.store import init_database, SchemaMismatchError
from raggen.core.config.project import default_project_config


def make_cfg(tmp_path, db_path: Path, **kwargs):
    cfg = default_project_config(tmp_path)
    cfg.storage.database_url = f"sqlite:///{db_path.resolve().as_posix()}"
    cfg.embedding.model_id = kwargs.get("embedding_model_id", "bge-small-en")
    cfg.embedding.dim = kwargs.get("embedding_dim", 1536)
    cfg.embedding.normalize = kwargs.get("embedding_normalized", True)
    cfg.notes = kwargs.get("notes", {})
    return cfg


def test_init_creates_tables_and_project_row(tmp_path):
    db_file = tmp_path / "rag.db"
    cfg = make_cfg(tmp_path, db_file)
    engine = init_database(cfg)
    insp = inspect(engine)
    assert "rag_project" in insp.get_table_names()
    assert "documents" in insp.get_table_names()
    assert "chunks" in insp.get_table_names()
    assert "embeddings" in insp.get_table_names()

    # verify project row
    with engine.connect() as conn:
        row = conn.execute(
            "SELECT * FROM rag_project WHERE id=1").mappings().fetchone()
        assert row is not None
        assert row["embedding_model_id"] == cfg.embedding.model_id
        assert int(row["embedding_dim"]) == cfg.embedding.dim


def test_init_validates_existing_config(tmp_path):
    db_file = tmp_path / "rag.db"
    cfg1 = make_cfg(tmp_path, db_file, embedding_dim=1234)
    init_database(cfg1)
    # same config should be fine
    init_database(cfg1)
    # different config should raise
    cfg2 = make_cfg(tmp_path, db_file, embedding_dim=9999)
    with pytest.raises(SchemaMismatchError):
        init_database(cfg2)


def test_destructive_reinit_allows_new_config(tmp_path):
    db_file = tmp_path / "rag.db"
    cfg1 = make_cfg(tmp_path, db_file, embedding_dim=1234)
    init_database(cfg1)
    cfg2 = make_cfg(tmp_path, db_file, embedding_dim=9999)
    init_database(cfg2, destructive=True)
    # verify now matches cfg2
    engine = init_database(cfg2)
    with engine.connect() as conn:
        row = conn.execute(
            "SELECT embedding_dim FROM rag_project WHERE id=1").mappings().fetchone()
        assert row and int(row["embedding_dim"]) == cfg2.embedding.dim
