from __future__ import annotations

from sqlalchemy.engine import Engine
from typing import Optional

_engine: Optional[Engine] = None


def set_engine(engine: Engine) -> None:
    global _engine
    _engine = engine


def get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("Engine not initialized. Call bootstrap() first.")
    return _engine
