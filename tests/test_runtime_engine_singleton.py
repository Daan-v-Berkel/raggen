from pathlib import Path

from raggen.core.bootstrap import bootstrap
from raggen.core.runtime import get_engine, set_engine
from raggen.core.config.project import default_project_config


def test_engine_singleton(tmp_path):
    # prepare minimal config file
    cfg_dir = tmp_path / ".rag"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "config.toml"
    # write minimal toml with only storage.database_url; loader should fill defaults
    db_path = tmp_path / "rag.db"
    cfg_file.write_text(f"[storage]\ndatabase_url = \"sqlite:///{db_path.resolve().as_posix()}\"\n")

    # call bootstrap which should create and register the engine
    cfg = bootstrap(cfg_file)

    e1 = get_engine()
    e2 = get_engine()
    assert e1 is e2

    # cleanup: unset engine
    set_engine(None)
