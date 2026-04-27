from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from pathlib import Path


def create_engine_from_url(database_url: str) -> Engine:
    url = make_url(database_url)

    # SQLite file databases: ensure the parent directory exists before connecting.
    # Other backends (postgres, etc.) manage their own connection setup.
    if url.drivername.startswith("sqlite"):
        database = url.database
        if database and database != ":memory:":
            p = Path(database)
            if not p.parent.exists():
                p.parent.mkdir(parents=True, exist_ok=True)

    return create_engine(database_url)
