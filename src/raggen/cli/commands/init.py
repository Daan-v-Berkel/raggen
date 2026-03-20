from __future__ import annotations

from raggen.core.init import do_init
from raggen.core.results import get_renderer
from raggen.core.results import project_result
from raggen.core.results.formats import OutputFormat


def run_init(
    *,
    root: str = ".",
    force: bool = False,
    format_as: OutputFormat = OutputFormat.JSON,
    detailed: bool = False,
) -> int:
    result = do_init(
        root=root,
        force=force,
    )

    projected = project_result(result, detail=detailed)
    renderer = get_renderer(format_as)
    print(renderer.render(projected))

    return 0 if result.success else 1
