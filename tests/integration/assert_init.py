from __future__ import annotations

from pathlib import Path

from tests.integration.helpers import (
    load_json,
    rag_config_path,
    rag_metadata_dir,
    rag_project_state_path,
)


def assert_init_layout(root: Path) -> None:
    rag_dir = rag_config_path(root).parent
    metadata_dir = rag_metadata_dir(root)
    config_path = rag_config_path(root)
    state_path = rag_project_state_path(root)

    assert rag_dir.is_dir(), f"Expected directory to exist: {rag_dir}"
    assert metadata_dir.is_dir(
    ), f"Expected directory to exist: {metadata_dir}"
    assert config_path.is_file(), f"Expected file to exist: {config_path}"
    assert state_path.is_file(), f"Expected file to exist: {state_path}"


def assert_init_state(root: Path) -> None:
    state_path = rag_project_state_path(root)
    state = load_json(state_path)

    foundation = state.get("foundation", {})
    assert foundation.get("project_root") == str(root.resolve())
    assert foundation.get("schema_version") == "v1"

    assert "updated_at" in state
    assert isinstance(state["updated_at"], str)
    assert "T" in state["updated_at"], f"Unexpected updated_at: {state['updated_at']}"

    assert state.get("state") == "initialised"
