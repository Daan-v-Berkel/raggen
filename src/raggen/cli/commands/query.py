from __future__ import annotations

from pathlib import Path
from textwrap import shorten

from raggen.core.bootstrap import bootstrap
from raggen.core.query.models import QueryRequest
from raggen.core.query.service import query
from raggen.core.results.formats import OutputFormat
from raggen.core.results.renderers import get_renderer


def run_query(
    text: str,
    *,
    config_path: str | None = None,
    top_k: int | None = None,
    format_as: OutputFormat = OutputFormat.JSON,
) -> int:
    """
    Run a retrieval query and print readable results.

    Returns process-style exit code:
      0 = success
      1 = error / no query text
    """
    if not text or not text.strip():
        print("Query text must not be empty.")
        return 1

    bootstrap(Path(config_path) if config_path else None)

    request = QueryRequest(
        text=text,
        top_k=top_k or 8,
    )

    result = query(request)
    renderer = get_renderer(format_as)

    print("Ingest finished:\n\n", renderer.render(result))

    return 0


def _format_snippet(text: str, width: int = 160) -> str:
    cleaned = " ".join(text.split())
    return shorten(cleaned, width=width, placeholder="...")
