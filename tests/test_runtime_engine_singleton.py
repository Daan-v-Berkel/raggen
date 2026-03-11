from raggen.core.bootstrap import bootstrap
from raggen.core.runtime import get_engine


def test_engine_singleton(
    tmp_path,
    cfg_factory,
    write_cfg,
    noop_backend_import,
):
    cfg = cfg_factory(tmp_path)
    cfg.storage.backend_key = "dummy_init"
    cfg.storage.vector_backend_import = noop_backend_import

    cfg_path = write_cfg(cfg, tmp_path)
    bootstrap(cfg_path)

    e1 = get_engine()
    e2 = get_engine()

    assert e1 is e2
