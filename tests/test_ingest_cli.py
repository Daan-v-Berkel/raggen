from pathlib import Path
import sys
import types
import os
import pytest

from raggen.core.ingest.config import default_project_config, save_project_config, load_project_config
from raggen.cli.commands.init import run_init


def test_save_and_load_toml_roundtrip(tmp_path):
    cfg = default_project_config(tmp_path)
    path = tmp_path / '.rag' / 'config.toml'
    save_project_config(cfg, path)
    loaded = load_project_config(path)
    assert loaded.embedding.model_id == cfg.embedding.model_id
    assert loaded.storage.database_url == cfg.storage.database_url


def test_cli_init_writes_config(tmp_path, monkeypatch):
    # run non-interactive init
    root = tmp_path
    run_init(root=str(root), non_interactive=True, destructive=False)
    p = root / '.rag' / 'config.toml'
    assert p.exists()
    loaded = load_project_config(p)
    assert loaded.project_root == root
