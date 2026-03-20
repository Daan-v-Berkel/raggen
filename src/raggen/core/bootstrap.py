from __future__ import annotations

from pathlib import Path
from typing import Optional
from tomlkit.exceptions import ParseError

from raggen.core.config.project import ProjectConfig
from raggen.core.config.config_utils import validate_file_groups
from raggen.core.runtime import (
    clear_runtime,
    get_config_path,
    is_initialized,
    set_runtime,
)
from raggen.core.store.engine import create_engine_from_url


DEFAULT_CONFIG_PATH = Path(".rag/config.toml")


class BootstrapError(RuntimeError):
    pass


def bootstrap(config_path: Optional[Path] = None) -> ProjectConfig:
    """
    Initialize and verify global application configuration + runtime.

    Rules:
    - bootstrap is single-shot per process
    - calling bootstrap twice with the same config path is a no-op
    - calling bootstrap twice with a different config path is an error
    - if bootstrap fails midway, global runtime/config are cleaned up
    """
    path = _resolve_config_path(config_path)

    # Already initialized: only allow exact same config path.
    if is_initialized():
        current_path = get_config_path()
        if current_path == path:
            cfg = ProjectConfig.get_config()
            if cfg is None:
                raise BootstrapError(
                    "Runtime says it is initialized, but ProjectConfig is missing."
                )
            return cfg

        raise BootstrapError(
            f"Application already bootstrapped from {current_path}. "
            f"Refusing to re-bootstrap from {path}."
        )

    if not path.exists():
        raise BootstrapError(
            f"Config file not found: {path}\n"
            "Run `rag init` to scaffold the project."
        )

    # Defensive check for partial state drift.
    existing_cfg = ProjectConfig.get_config()
    if existing_cfg is not None:
        raise BootstrapError(
            "ProjectConfig is already loaded, but runtime is not initialized. "
            "This indicates partial state. Clear config/runtime before bootstrapping again."
        )

    engine = None
    try:
        cfg = ProjectConfig.load_config(path)
    except ParseError as err:
        raise BootstrapError(f"Invalid config file: {path}\n{err}") from err
    try:
        validate_file_groups(cfg)

        engine = create_engine_from_url(cfg.storage.database_url)

        # Verify engine is actually usable before publishing it globally.
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")

        set_runtime(engine=engine, config_path=path)
        return cfg

    except Exception:
        # Prevent partial initialization from lingering.
        try:
            ProjectConfig.clear_config()
        except Exception:
            pass

        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                pass

        clear_runtime(dispose_engine=False)
        raise


def _resolve_config_path(config_path: Optional[Path]) -> Path:
    if config_path is not None:
        return Path(config_path).expanduser().resolve()
    return DEFAULT_CONFIG_PATH.resolve()
