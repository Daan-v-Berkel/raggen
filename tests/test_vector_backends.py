import importlib
import types
import json
from pathlib import Path
import pytest
from sqlalchemy import inspect
from sqlalchemy import create_engine

from raggen.core.store.plugin_loader import load_object, load_vector_backend
from raggen.core.store.init_config import RagInitConfig
from raggen.core.store.initializer import init_database


class DummyBackend:
    def __init__(self):
        self.key = "dummy"
        self.created = False
        self.dropped = False

    def supports(self, engine):
        return True

    def create_schema(self, engine, dim):
        self.created = True

    def drop_schema(self, engine):
        self.dropped = True

    def upsert_vectors(self, engine, *, vectors, embedding_model_id, dim, normalized):
        # record a simple validation
        for cid, v in vectors:
            if len(v) != dim:
                raise ValueError("bad dim")


def test_load_vector_backend_import_path_success(tmp_path, monkeypatch):
    # create a fake module path by injecting into sys.modules
    mod = types.ModuleType("tests._dummy_mod")
    mod.DummyBackend = DummyBackend
    import sys

    sys.modules["tests._dummy_mod"] = mod
    inst = load_vector_backend("tests._dummy_mod:DummyBackend")
    assert isinstance(inst, DummyBackend)


def test_initializer_calls_backend_create_schema(tmp_path, monkeypatch):
    db_file = tmp_path / "rag.db"
    cfg = RagInitConfig(
        database_url=f"sqlite:///{db_file}",
        embedding_model_id="x",
        embedding_dim=42,
        embedding_normalized=True,
        backend_key="dummy",
        vector_backend_import="tests._dummy_mod:DummyBackend",
    )
    # ensure our DummyBackend module is available
    import types, sys
    mod = types.ModuleType("tests._dummy_mod")
    mod.DummyBackend = DummyBackend
    sys.modules["tests._dummy_mod"] = mod

    engine = init_database(cfg)
    backend = getattr(engine, "_rag_vector_backend", None)
    assert backend is not None and backend.created is True


def test_destructive_reinit_calls_drop_schema(tmp_path):
    db_file = tmp_path / "rag.db"
    cfg = RagInitConfig(
        database_url=f"sqlite:///{db_file}",
        embedding_model_id="x",
        embedding_dim=16,
        embedding_normalized=True,
        backend_key="dummy",
        vector_backend_import="tests._dummy_mod:DummyBackend",
    )
    import types, sys
    mod = types.ModuleType("tests._dummy_mod")
    mod.DummyBackend = DummyBackend
    sys.modules["tests._dummy_mod"] = mod

    engine = init_database(cfg)
    backend = getattr(engine, "_rag_vector_backend", None)
    assert backend is not None and backend.created is True
    engine = init_database(cfg, destructive=True)
    backend = getattr(engine, "_rag_vector_backend", None)
    assert backend is not None and backend.dropped is True
