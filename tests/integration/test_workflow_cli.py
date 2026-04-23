from __future__ import annotations

from pathlib import Path
import pytest

from tests.integration.helpers import build_command, run_cli, rag_config_path, rag_project_state_path
from tests.integration.assert_init import assert_init_layout, assert_init_state


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

    # Extension points (placeholders): future steps will look like:
    # result = run_cli("build", str(project_root))
    # assert result.returncode == 0
    # assert_build(project_root)

    # result = run_cli("ingest", str(project_root))
    # assert result.returncode == 0
    # assert_ingest(project_root)

    # result = run_cli("runs", str(project_root))
    # assert result.returncode == 0
    # assert_runs(project_root)
