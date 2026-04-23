from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests.integration.helpers import (
    load_json,
    rag_dir,
    rag_project_state_path,
)


def assert_build_layout(root: Path) -> None:
    db_path = rag_dir(root) / "rag.db"
    assert db_path.is_file(), f"Expected database file to exist: {db_path}"


def assert_build_state(root: Path) -> None:
    state_path = rag_project_state_path(root)
    state = load_json(state_path)
    assert state.get("state") == "set_up", (
        f"Expected project state 'set_up', got: {state.get('state')}"
    )

    updated_at = datetime.fromisoformat(state["updated_at"])
    threshold = datetime.now(timezone.utc) - timedelta(minutes=2)
    assert updated_at >= threshold, (
        f"State timestamp is too old: {state['updated_at']}"
    )
