from __future__ import annotations

from pathlib import Path
from typing import Optional

from raggen.core.config.project import ProjectConfig


DEFAULT_CONFIG_PATH = Path(".rag/config.toml")


class BootstrapError(RuntimeError):
    pass


def bootstrap(config_path: Optional[Path] = None) -> ProjectConfig:
    """
    Initialize global application configuration.

    This should be called once at application startup

    Args:
        config_path: Optional explicit config path.

    Returns:
        Loaded ProjectConfig.
    """

    path = _resolve_config_path(config_path)

    if not path.exists():
        raise BootstrapError(
            f"Config file not found: {path}\n"
            "Run `rag init` to initialize the project."
        )

    cfg = ProjectConfig.load_config(path)

    return cfg


def _resolve_config_path(config_path: Optional[Path]) -> Path:
    """
    Resolve config path using explicit path or default location.
    """
    if config_path:
        return Path(config_path).expanduser().resolve()

    return DEFAULT_CONFIG_PATH.resolve()
