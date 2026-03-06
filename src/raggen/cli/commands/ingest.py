from __future__ import annotations

from pathlib import Path
from raggen.core.config.project import ProjectConfig
from raggen.core.ingest.ingest_service import ingest_only


def run_ingest(*, config_path: str = '.rag/config.toml') -> None:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"Config not found at {cfg_path}")
        return
    cfg = ProjectConfig.load_config(cfg_path)
    stats = ingest_only(cfg=cfg)
    print("Ingest finished:", stats)
