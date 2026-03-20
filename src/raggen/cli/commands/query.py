from __future__ import annotations

from pathlib import Path
from textwrap import shorten

from raggen.core.bootstrap import bootstrap, BootstrapError
from raggen.core.query.models import QueryRequest
from raggen.core.query.service import query
from raggen.core.results.formats import OutputFormat
from raggen.core.results.renderers import get_renderer
from raggen.core.results.projection import project_result


def run_query(
    text: str,
    *,
    config_path: str | None = None,
    top_k: int | None = None,
    format_as: OutputFormat = OutputFormat.JSON,
    detailed: bool = False,
) -> int:
    """
    Run a retrieval query and print readable results.

    Returns process-style exit code:
      0 = success
      1 = error / no query text
    """
    try:
        bootstrap(Path(config_path) if config_path else None)
    except BootstrapError as e:
        print(f"Error: {e}")
        return 1

    if not text or not text.strip():
        print("Query text must not be empty.")
        return 1

    request = QueryRequest(
        text=text,
        top_k=top_k or 8,
    )

    result = query(request)
    projected = project_result(result, detailed=detailed)
    renderer = get_renderer(format_as)

    print("Ingest finished:\n\n", renderer.render(projected))

    return 0


def _format_snippet(text: str, width: int = 160) -> str:
    cleaned = " ".join(text.split())
    return shorten(cleaned, width=width, placeholder="...")
