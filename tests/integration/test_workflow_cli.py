from __future__ import annotations

from pathlib import Path
import pytest

from tests.integration.helpers import run_cli
from tests.integration.assert_init import assert_init_layout, assert_init_state
from tests.integration.assert_build import assert_build_layout, assert_build_state
from tests.integration.assert_ingest import assert_ingest_layout, assert_ingest_state


@pytest.mark.integration
def test_workflow_init(tmp_path: Path) -> None:
    """
    Workflow-style integration test scaffold.

    Steps are explicit here: future command steps should be added in order
    and have corresponding assert_<command>() helpers in tests/integration.
    """
    project_root = tmp_path / "project"
    project_root.mkdir()

    # Run `rag init <root>`
    result = run_cli("init", str(project_root))
    assert result.returncode == 0, (
        f"CLI failed unexpectedly.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    # Assert filesystem layout and persistent state
    assert_init_layout(project_root)
    assert_init_state(project_root)

    # Run `rag build` from within the project root
    result = run_cli("build", cwd=project_root)
    assert result.returncode == 0, (
        f"CLI failed unexpectedly.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    assert_build_layout(project_root)
    assert_build_state(project_root)

    # Create files to index
    (project_root / "hello.txt").write_text("Hello world. This is a test document.")
    (project_root / "notes").mkdir()
    (project_root / "notes" / "readme.txt").write_text("Project notes. Important information here.")

    # Create a file ignored via .gitignore (tests scan.ignore_files behaviour)
    (project_root / ".gitignore").write_text("ignored_file.txt\n")
    (project_root / "ignored_file.txt").write_text("This file should not be indexed.")

    # Run `rag ingest` from within the project root
    result = run_cli("ingest", cwd=project_root)
    assert result.returncode == 0, (
        f"CLI failed unexpectedly.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    assert_ingest_layout(project_root)
    assert_ingest_state(
        project_root,
        expected_docs=["hello.txt", "notes/readme.txt"],
        unexpected_docs=["ignored_file.txt"],
    )

    # Extension points (placeholders): future steps will look like:
    # result = run_cli("runs", str(project_root))
    # assert result.returncode == 0
    # assert_runs(project_root)
