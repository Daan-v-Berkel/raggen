from __future__ import annotations

from pathlib import Path
from raggen.core.bootstrap import bootstrap
from raggen.core.ingest.ingest_service import do_ingest
from raggen.core.results.formats import OutputFormat
from raggen.core.results.renderers import get_renderer


def run_ingest(
    *, config_path: str = ".rag/config.toml",
    destructive: bool = False,
    format_as: OutputFormat = OutputFormat.JSON
) -> None:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"Config not found at {cfg_path}")
        return
    _ = bootstrap(cfg_path)
    result = do_ingest(destructive=destructive)
    renderer = get_renderer(format_as)
    print("Ingest finished:\n\n", renderer.render(result))
