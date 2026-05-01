from __future__ import annotations

from pathlib import Path
from raggen.core.bootstrap import bootstrap, BootstrapError
from raggen.core.build.build_service import do_build
from raggen.core.results import get_renderer
from raggen.core.results import project_result
from raggen.core.results.formats import OutputFormat


def run_build(
    *,
    config: str = ".rag/config.toml",
    destructive: bool = False,
    format_as: OutputFormat = OutputFormat.JSON,
    detailed: bool = False,
) -> int:
    try:
        bootstrap(Path(config) if config else None)
    except BootstrapError as e:
        print(f"Error: {e}")
        return 1

    result = do_build(
        config_path=config,
        destructive=destructive,
    )

    projected = project_result(result, detailed=detailed)
    renderer = get_renderer(format_as)
    print(renderer.render(projected))

    return 0 if result.success else 1
