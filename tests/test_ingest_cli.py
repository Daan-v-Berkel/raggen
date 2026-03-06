from raggen.core.ingest.config import ProjectConfig
from raggen.cli.commands.init import run_init


def test_save_and_load_toml_roundtrip(tmp_path):
    run_init(root=tmp_path, non_interactive=True, destructive=False)
    p = tmp_path / '.rag' / 'config.toml'
    assert p.exists()
    loaded = ProjectConfig.load_config(p)
    assert loaded.project_root == tmp_path
    loaded.schema_version = 'v2'
    loaded.save(p)

    cfg = ProjectConfig.load_config(p)
    assert loaded.embedding.model_id == cfg.embedding.model_id
    assert loaded.storage.database_url == cfg.storage.database_url
