from raggen.core.bootstrap import bootstrap
from raggen.core.store.initializer import init_database
from raggen.core.store.plugin_loader import load_vector_backend


def test_load_vector_backend_import_path_success(tracking_backend_import):
    import_path, backend_cls = tracking_backend_import

    inst = load_vector_backend(import_path)

    assert isinstance(inst, backend_cls)


def test_initializer_calls_backend_create_schema(
    tmp_path,
    cfg_factory,
    write_cfg,
    tracking_backend_import,
):
    import_path, backend_cls = tracking_backend_import

    cfg = cfg_factory(tmp_path)
    cfg.storage.backend_key = "dummy"
    cfg.storage.vector_backend_import = import_path
    cfg.embedding.model_id = "x"
    cfg.embedding.dim = 42
    cfg.embedding.normalize = True

    cfg_path = write_cfg(cfg, tmp_path)
    cfg = bootstrap(cfg_path)

    engine = init_database(cfg)

    backend = getattr(engine, "_rag_vector_backend", None)
    assert backend is not None

    # verify schema creation was invoked with the configured dim
    assert backend_cls.created_calls == [42]


def test_destructive_reinit_calls_drop_schema(
    tmp_path,
    cfg_factory,
    write_cfg,
    tracking_backend_import,
):
    import_path, backend_cls = tracking_backend_import

    cfg = cfg_factory(tmp_path)
    cfg.storage.backend_key = "dummy"
    cfg.storage.vector_backend_import = import_path
    cfg.embedding.model_id = "x"
    cfg.embedding.dim = 16
    cfg.embedding.normalize = True

    cfg_path = write_cfg(cfg, tmp_path)
    cfg = bootstrap(cfg_path)

    engine = init_database(cfg)
    backend = getattr(engine, "_rag_vector_backend", None)
    assert backend is not None

    # initial init should create schema once
    assert backend_cls.created_calls == [16]
    assert backend_cls.dropped_calls == []

    engine = init_database(cfg, destructive=True)
    backend = getattr(engine, "_rag_vector_backend", None)
    assert backend is not None

    # destructive reinit should drop, then recreate
    assert backend_cls.dropped_calls == [True]
    assert backend_cls.created_calls == [16, 16]
