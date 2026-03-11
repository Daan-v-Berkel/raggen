from __future__ import annotations

from pathlib import Path
from typing import Optional

from sqlalchemy.engine import Engine

_engine: Optional[Engine] = None
_config_path: Optional[Path] = None


def set_runtime(*, engine: Engine, config_path: Path) -> None:
    global _engine, _config_path

    if _engine is not None or _config_path is not None:
        raise RuntimeError("Runtime already initialized.")

    _engine = engine
    _config_path = config_path.resolve()


def get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("Engine not initialized. Call bootstrap() first.")
    return _engine


def get_config_path() -> Path:
    if _config_path is None:
        raise RuntimeError("Runtime not initialized. Call bootstrap() first.")
    return _config_path


def is_initialized() -> bool:
    return _engine is not None and _config_path is not None


def clear_runtime(*, dispose_engine: bool = True) -> None:
    global _engine, _config_path

    if dispose_engine and _engine is not None:
        try:
            _engine.dispose()
        except Exception:
            pass

    _engine = None
    _config_path = None
