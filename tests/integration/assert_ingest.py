from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests.integration.helpers import (
    load_json,
    rag_dir,
)


def _latest_run_dir(root: Path) -> Path:
    runs_root = rag_dir(root) / "runs"
    run_dirs = sorted(d for d in runs_root.iterdir() if d.is_dir())
    assert run_dirs, f"No run directories found in: {runs_root}"
    return run_dirs[-1]


def assert_ingest_layout(root: Path) -> None:
    runs_root = rag_dir(root) / "runs"
    assert runs_root.is_dir(), f"Expected runs directory to exist: {runs_root}"

    run_dir = _latest_run_dir(root)
    result_path = run_dir / "result.json"
    assert result_path.is_file(), f"Expected result file to exist: {result_path}"


def assert_ingest_state(
    root: Path,
    expected_docs: list[str],
    unexpected_docs: list[str],
) -> None:
    run_dir = _latest_run_dir(root)
    result = load_json(run_dir / "result.json")

    assert result.get("operation") == "ingest", (
        f"Expected operation 'ingest', got: {result.get('operation')}"
    )
    assert result.get("success") is True, (
        f"Expected success=true, got: {result.get('success')}\n"
        f"Errors: {result.get('errors')}\nWarnings: {result.get('warnings')}"
    )

    summary = result.get("data", {}).get("summary", {})
    assert summary.get("docs_parsed") >= len(expected_docs), (
        f"Expected at least {len(expected_docs)} docs_parsed, got: {summary.get('docs_parsed')}"
    )

    created_at = datetime.fromisoformat(result["created_at"])
    threshold = datetime.now(timezone.utc) - timedelta(minutes=2)
    assert created_at >= threshold, (
        f"Run timestamp is too old: {result['created_at']}"
    )

    db_path = rag_dir(root) / "rag.db"
    with sqlite3.connect(str(db_path)) as conn:
        indexed = {
            row[0]
            for row in conn.execute("SELECT doc_id FROM documents").fetchall()
        }

    for doc in expected_docs:
        assert doc in indexed, f"Expected document to be indexed: {doc}"

    for doc in unexpected_docs:
        assert doc not in indexed, f"Expected document to NOT be indexed: {doc}"

    with sqlite3.connect(str(db_path)) as conn:
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert chunk_count >= len(expected_docs), (
        f"Expected at least {len(expected_docs)} chunks, got: {chunk_count}"
    )
