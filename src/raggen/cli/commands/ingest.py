from __future__ import annotations

from pathlib import Path
from ...core.ingest.config import load_project_config
from ...core.ingest.ingest_service import ingest_only


def run_ingest(*, config_path: str = '.rag/config.toml') -> None:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"Config not found at {cfg_path}")
        return
    cfg = load_project_config(cfg_path)
    stats = ingest_only(cfg=cfg)
    print("Ingest finished:", stats)
