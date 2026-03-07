import types

from raggen.core.store.plugin_loader import load_vector_backend
from raggen.core.store.initializer import init_database
from raggen.core.config.project import default_project_config
from raggen.core.store.engine import create_engine_from_url
from raggen.core.runtime import set_engine


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
    cfg = default_project_config(tmp_path)

    cfg.storage.database_url = f"sqlite:///{db_file.resolve().as_posix()}"
    cfg.storage.backend_key = "dummy"
    cfg.storage.vector_backend_import = "tests._dummy_mod:DummyBackend"
    cfg.embedding.model_id = "x"
    cfg.embedding.dim = 42
    cfg.embedding.normalize = True

    # ensure our DummyBackend module is available
    import types
    import sys
    mod = types.ModuleType("tests._dummy_mod")
    mod.DummyBackend = DummyBackend
    sys.modules["tests._dummy_mod"] = mod

    engine = init_database(cfg)
    backend = getattr(engine, "_rag_vector_backend", None)
    assert backend is not None


def test_destructive_reinit_calls_drop_schema(tmp_path):
    db_file = tmp_path / "rag.db"
    cfg = default_project_config(tmp_path)

    cfg.storage.database_url = f"sqlite:///{db_file.resolve().as_posix()}"
    cfg.storage.backend_key = "dummy"
    cfg.storage.vector_backend_import = "tests._dummy_mod:DummyBackend"
    cfg.embedding.model_id = "x"
    cfg.embedding.dim = 16
    cfg.embedding.normalize = True

    import types
    import sys
    mod = types.ModuleType("tests._dummy_mod")
    mod.DummyBackend = DummyBackend
    sys.modules["tests._dummy_mod"] = mod

    # write config and bootstrap to ensure runtime engine is registered
    cfg_dir = tmp_path / '.rag'
    cfg_dir.mkdir(exist_ok=True)
    cfg_file = cfg_dir / 'config.toml'
    cfg_file.write_text(
        f"[storage]\ndatabase_url = \"{cfg.storage.database_url}\"\n")
    from raggen.core.bootstrap import bootstrap
    cfg = bootstrap(cfg_file)

    engine = init_database(cfg)
    backend = getattr(engine, "_rag_vector_backend", None)
    assert backend is not None
    engine = init_database(cfg, destructive=True)
    backend = getattr(engine, "_rag_vector_backend", None)
    assert backend is not None
