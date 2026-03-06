from __future__ import annotations

from pathlib import Path
from raggen.core.bootstrap import bootstrap
from raggen.core.ingest.ingest_service import do_ingest


def run_ingest(*, config_path: str = '.rag/config.toml', destructive: bool = False) -> None:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"Config not found at {cfg_path}")
        return
    _ = bootstrap(cfg_path)
    stats = do_ingest(destructive=destructive)
    print("Ingest finished:", stats)
