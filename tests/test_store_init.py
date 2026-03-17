from sqlalchemy import inspect, text
import pytest

from raggen.core.store import init_database, SchemaMismatchError
from raggen.core.bootstrap import bootstrap


def test_init_creates_tables_and_project_row(tmp_path, cfg_factory, write_cfg):
    cfg = cfg_factory(tmp_path)
    cfg_path = write_cfg(cfg, tmp_path)

    cfg = bootstrap(cfg_path)
    engine = init_database(cfg)

    insp = inspect(engine)
    assert "rag_project" in insp.get_table_names()
    assert "documents" in insp.get_table_names()
    assert "chunks" in insp.get_table_names()
    assert "embeddings" in insp.get_table_names()

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM rag_project WHERE id=1")
        ).mappings().fetchone()

        assert row is not None
        assert row["embedding_model_id"] == cfg.embedding.model_id
        assert int(row["embedding_dim"]) == cfg.embedding.dim


def test_init_validates_existing_config(tmp_path, cfg_factory, write_cfg):
    cfg1 = cfg_factory(tmp_path, embedding_dim=1234)
    cfg_path = write_cfg(cfg1, tmp_path)

    cfg1 = bootstrap(cfg_path)
    init_database(cfg1)
    init_database(cfg1)

    cfg2 = cfg_factory(tmp_path, embedding_dim=9999)

    with pytest.raises(SchemaMismatchError):
        init_database(cfg2)


def test_destructive_reinit_allows_new_config(tmp_path, cfg_factory, write_cfg):
    cfg1 = cfg_factory(tmp_path, embedding_dim=1234)
    cfg_path = write_cfg(cfg1, tmp_path)

    cfg1 = bootstrap(cfg_path)
    init_database(cfg1)

    cfg2 = cfg_factory(tmp_path, embedding_dim=9999)
    init_database(cfg2, destructive=True)

    engine = init_database(cfg2)
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT embedding_dim FROM rag_project WHERE id=1")
        ).mappings().fetchone()

        assert row is not None
        assert int(row["embedding_dim"]) == cfg2.embedding.dim
