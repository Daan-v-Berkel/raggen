from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest


# You can override this in CI if needed:
#   RAG_TEST_CLI="python -m raggen.cli"
#
# By default, these tests use the installed console script: `rag`
CLI_COMMAND = os.environ.get("RAG_TEST_CLI", "rag")


def _build_command(*args: str) -> list[str]:
    """
    Build the command to execute the CLI.

    Supports either:
      - a plain executable name, e.g. "rag"
      - a multi-part command via env var, e.g. "python -m raggen.cli"
    """
    return CLI_COMMAND.split() + list(args)


def _run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """
    Run the real CLI as a subprocess and capture output.
    """
    return subprocess.run(
        _build_command(*args),
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_rag_tree_exists(root: Path) -> tuple[Path, Path, Path, Path]:
    """
    Assert the expected init tree exists and return the main paths.
    """
    rag_dir = root / ".rag"
    metadata_dir = rag_dir / "metadata"
    config_path = rag_dir / "config.toml"
    state_path = metadata_dir / "project_state.json"

    assert rag_dir.is_dir(), f"Expected directory to exist: {rag_dir}"
    assert metadata_dir.is_dir(
    ), f"Expected directory to exist: {metadata_dir}"
    assert config_path.is_file(), f"Expected file to exist: {config_path}"
    assert state_path.is_file(), f"Expected file to exist: {state_path}"

    return rag_dir, metadata_dir, config_path, state_path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _assert_nonempty_file(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    assert content.strip(), f"Expected non-empty file: {path}"


def _assert_project_state(
    state: dict,
    *,
    root: Path,
    force: bool,
) -> None:
    """
    Assert the stable/public contract of project_state.json for init.
    """
    foundation = state["foundation"]
    assert foundation["project_root"] == str(root.resolve())
    assert foundation["schema_version"] == "v1"

    assert "updated_at" in state
    assert isinstance(state["updated_at"], str)
    # Keep this loose enough to avoid brittleness, but strong enough to catch junk.
    assert "T" in state["updated_at"], f"Unexpected updated_at: {state['updated_at']}"

    assert state["state"] == "initialised"


@pytest.mark.integration
def test_init_creates_expected_project_structure(tmp_path: Path) -> None:
    """
    Integration test:
    `rag init <root>` should create the expected .rag tree and metadata.
    """
    project_root = tmp_path / "project"
    project_root.mkdir()

    result = _run_cli("init", str(project_root))

    assert result.returncode == 0, (
        f"CLI failed unexpectedly.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    _, _, config_path, state_path = _assert_rag_tree_exists(project_root)

    _assert_nonempty_file(config_path)

    state = _load_json(state_path)
    _assert_project_state(state, root=project_root, force=False)


@pytest.mark.integration
def test_init_force_replaces_existing_rag_directory(tmp_path: Path) -> None:
    """
    Integration test:
    `rag init <root> --force` should remove an existing .rag directory
    and recreate it cleanly.
    """
    project_root = tmp_path / "project"
    rag_dir = project_root / ".rag"
    metadata_dir = rag_dir / "metadata"

    project_root.mkdir(parents=True)

    # Create a pre-existing .rag tree with junk/old data that should disappear.
    metadata_dir.mkdir(parents=True)
    (rag_dir / "config.toml").write_text("old_config = true\n", encoding="utf-8")
    (metadata_dir / "project_state.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "operation": "init",
                "success": True,
                "data": {"state": "old"},
                "warnings": [],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    junk_file = rag_dir / "should_be_removed.txt"
    junk_file.write_text("delete me", encoding="utf-8")

    result = _run_cli("init", str(project_root), "--force")

    assert result.returncode == 0, (
        f"CLI failed unexpectedly.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    _, _, config_path, state_path = _assert_rag_tree_exists(project_root)

    # The old junk file should be gone if .rag was actually removed/recreated.
    assert not junk_file.exists(), "Expected pre-existing .rag contents to be removed"

    _assert_nonempty_file(config_path)

    state = _load_json(state_path)
    _assert_project_state(state, root=project_root, force=True)


@pytest.mark.integration
def test_init_force_on_missing_rag_still_initialises_normally(tmp_path: Path) -> None:
    """
    Integration test:
    `rag init <root> --force` should still work even if .rag does not exist yet.
    """
    project_root = tmp_path / "project"
    project_root.mkdir()

    result = _run_cli("init", str(project_root), "--force")

    assert result.returncode == 0, (
        f"CLI failed unexpectedly.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    _, _, config_path, state_path = _assert_rag_tree_exists(project_root)

    _assert_nonempty_file(config_path)

    state = _load_json(state_path)
    _assert_project_state(state, root=project_root, force=True)
