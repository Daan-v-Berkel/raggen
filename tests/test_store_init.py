from sqlalchemy import inspect, text
import pytest

from sqlalchemy import select

from raggen.core.store import init_database, SchemaMismatchError
from raggen.core.store.metadata_schema import rag_project
from raggen.core.bootstrap import bootstrap
from raggen.core.config.project import ProjectConfig


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


def test_embedding_normalized_round_trips_as_bool(tmp_path, cfg_factory, write_cfg):
    """embedding_normalized must be stored and read back as a Python bool.

    SQLite previously coerced True → 1 silently; PostgreSQL rejected it.
    The column is now typed Boolean so both backends return a bool on read.
    """
    cfg = cfg_factory(tmp_path, embedding_normalized=True)
    cfg_path = write_cfg(cfg, tmp_path)
    cfg = bootstrap(cfg_path)
    engine = init_database(cfg)

    # Use the typed table expression so SQLAlchemy's Boolean processor runs
    # and converts the stored value to a Python bool (not raw int).
    with engine.connect() as conn:
        row = conn.execute(
            select(rag_project).where(rag_project.c.id == 1)
        ).mappings().fetchone()

    assert row["embedding_normalized"] is True


def test_normalize_true_does_not_trigger_drift_on_second_build(
    tmp_path, cfg_factory, write_cfg
):
    """A second init_database call with the same config must not raise.

    Before the Boolean column fix, the bool→int normalisation in
    _compare_configs hid a latent mismatch that would surface on PostgreSQL.
    """
    cfg = cfg_factory(tmp_path, embedding_normalized=True)
    cfg_path = write_cfg(cfg, tmp_path)
    cfg = bootstrap(cfg_path)
    init_database(cfg)
    # must not raise SchemaMismatchError
    init_database(cfg)


def test_normalize_false_does_not_trigger_drift_on_second_build(
    tmp_path, cfg_factory, write_cfg
):
    cfg = cfg_factory(tmp_path, embedding_normalized=False)
    cfg_path = write_cfg(cfg, tmp_path)
    cfg = bootstrap(cfg_path)
    init_database(cfg)
    init_database(cfg)


def test_init_database_returns_plain_engine_without_rag_attribute(
    tmp_path, cfg_factory, write_cfg
):
    """init_database must not monkey-patch _rag_vector_backend onto the engine."""
    cfg = cfg_factory(tmp_path)
    cfg_path = write_cfg(cfg, tmp_path)
    cfg = bootstrap(cfg_path)
    engine = init_database(cfg)

    assert not hasattr(engine, "_rag_vector_backend")
