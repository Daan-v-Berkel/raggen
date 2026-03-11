import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def isolated_runtime():
    from raggen.core.runtime import clear_runtime
    from raggen.core.config.project import ProjectConfig

    ProjectConfig.clear_config()
    clear_runtime()

    yield

    ProjectConfig.clear_config()
    clear_runtime()


@pytest.fixture
def dummy_backend_import():
    import sys
    import types

    mod = types.ModuleType("tests._dummy_mod")

    class DummyBackend:
        key = "dummy"

        def supports(self, engine):
            return True

        def create_schema(self, engine, dim):
            pass

        def drop_schema(self, engine):
            pass

        def upsert_vectors(self, *args, **kwargs):
            return None

        def delete_vectors(self, *args, **kwargs):
            return None

        def search(self, *args, **kwargs):
            return []

    mod.DummyBackend = DummyBackend
    sys.modules["tests._dummy_mod"] = mod
    return "tests._dummy_mod:DummyBackend"


@pytest.fixture
def cfg_factory(dummy_backend_import):
    from raggen.core.config.project import default_project_config

    def _make(tmp_path: Path, db_name: str = "rag.db", **kwargs):
        db_file = tmp_path / db_name
        cfg = default_project_config(tmp_path)
        cfg.storage.database_url = f"sqlite:///{db_file.resolve().as_posix()}"
        cfg.storage.vector_backend_import = dummy_backend_import
        cfg.storage.backend_key = "dummy"
        cfg.embedding.model_id = kwargs.get(
            "embedding_model_id", "bge-small-en")
        cfg.embedding.dim = kwargs.get("embedding_dim", 1536)
        cfg.embedding.normalize = kwargs.get("embedding_normalized", True)
        return cfg

    return _make


@pytest.fixture
def write_cfg():
    def _write(cfg, root: Path):
        rag_dir = root / ".rag"
        rag_dir.mkdir(exist_ok=True)
        cfg_path = rag_dir / "config.toml"
        cfg.save(cfg_path)
        return cfg_path

    return _write


@pytest.fixture
def tracking_backend_import():
    import sys
    import types

    mod = types.ModuleType("tests._tracking_backend_mod")

    class TrackingBackend:
        key = "dummy"
        created_calls = []
        dropped_calls = []

        def __init__(self):
            pass

        def supports(self, engine):
            return True

        def create_schema(self, engine, dim):
            type(self).created_calls.append(dim)

        def drop_schema(self, engine):
            type(self).dropped_calls.append(True)

        def upsert_vectors(self, engine, *, vectors, embedding_model_id, dim, normalized):
            for cid, v in vectors:
                if len(v) != dim:
                    raise ValueError("bad dim")

        def delete_vectors(self, engine, *, chunks):
            return None

        def search(self, engine, *, query_vector, top_k):
            return []

    # reset class-level tracking for each test
    TrackingBackend.created_calls = []
    TrackingBackend.dropped_calls = []

    mod.TrackingBackend = TrackingBackend
    sys.modules["tests._tracking_backend_mod"] = mod

    return "tests._tracking_backend_mod:TrackingBackend", TrackingBackend


@pytest.fixture
def noop_backend_import():
    import sys
    import types

    mod = types.ModuleType("tests._noop_backend_mod")

    class NoopBackend:
        key = "dummy_init"

        def supports(self, engine):
            return True

        def create_schema(self, engine, dim):
            pass

        def drop_schema(self, engine):
            pass

        def upsert_vectors(self, *args, **kwargs):
            return None

        def delete_vectors(self, *args, **kwargs):
            return None

        def search(self, engine, *, query_vector, top_k):
            return []

    mod.NoopBackend = NoopBackend
    sys.modules["tests._noop_backend_mod"] = mod

    return "tests._noop_backend_mod:NoopBackend"
