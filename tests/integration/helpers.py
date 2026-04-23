from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

# You can override this in CI if needed:
#   RAG_TEST_CLI="python -m raggen.cli"
# By default, these tests use the installed console script: `rag`

CLI_COMMAND = os.environ.get("RAG_TEST_CLI", "rag")


def build_command(*args: str) -> list[str]:
    """
    Build the command to execute the CLI.

    Supports either:
      - a plain executable name, e.g. "rag"
      - a multi-part command via env var, e.g. "python -m raggen.cli"
    """
    return CLI_COMMAND.split() + list(args)


def run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """
    Run the real CLI as a subprocess and capture output.
    """
    return subprocess.run(
        build_command(*args),
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def assert_nonempty_file(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    assert content.strip(), f"Expected non-empty file: {path}"


# small path helpers for .rag

def rag_dir(root: Path) -> Path:
    return root / ".rag"


def rag_metadata_dir(root: Path) -> Path:
    return rag_dir(root) / "metadata"


def rag_config_path(root: Path) -> Path:
    return rag_dir(root) / "config.toml"


def rag_project_state_path(root: Path) -> Path:
    return rag_metadata_dir(root) / "project_state.json"


# Export names for direct import from tests.integration.helpers
__all__ = [
    "CLI_COMMAND",
    "build_command",
    "run_cli",
    "load_json",
    "assert_nonempty_file",
    "rag_dir",
    "rag_metadata_dir",
    "rag_config_path",
    "rag_project_state_path",
]
