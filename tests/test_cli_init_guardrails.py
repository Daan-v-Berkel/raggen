import pytest

from raggen.cli.commands.init import run_init


def test_cli_init_refuses_when_already_initialized(tmp_path):
    root = tmp_path
    rag_dir = root / ".rag"
    rag_dir.mkdir()
    cfg = rag_dir / "config.toml"
    cfg.write_text("[project]\nroot='.'\n")

    with pytest.raises(SystemExit):
        run_init(root=str(root), non_interactive=True, force=False)

    assert cfg.exists()


def test_cli_init_force_overwrites(tmp_path):
    root = tmp_path
    rag_dir = root / ".rag"
    rag_dir.mkdir()
    cfg = rag_dir / "config.toml"
    cfg.write_text("old")
    # run with force
    run_init(root=str(root), non_interactive=True, force=True)
    # After force, config should be rewritten
    assert cfg.exists()
    assert cfg.read_text() != "old"
