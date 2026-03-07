from __future__ import annotations

from pathlib import Path
from textwrap import shorten

from raggen.core.bootstrap import bootstrap
from raggen.core.query.models import QueryRequest
from raggen.core.query.service import query


def run_query(
    text: str,
    *,
    config_path: str | None = None,
    top_k: int | None = None,
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

    response = query(request)

    if not response.matches:
        print("No matches found.")
        return 0

    print(f'Query: "{response.query}"')
    print(f"Model: {response.used_query_model}")
    print()

    for i, match in enumerate(response.matches, start=1):
        snippet = _format_snippet(match.text)
        print(f"{i}. {match.doc_id}")
        print(f"   score: {match.score:.6f}")
        print(f"   {snippet}")
        print()

    return 0


def _format_snippet(text: str, width: int = 160) -> str:
    cleaned = " ".join(text.split())
    return shorten(cleaned, width=width, placeholder="...")
