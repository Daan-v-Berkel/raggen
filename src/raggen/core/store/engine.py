from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from pathlib import Path


def create_engine_from_url(database_url: str) -> Engine:
    url = make_url(database_url)

    # SQLite file databases: ensure the parent directory exists before connecting.
    if url.drivername.startswith("sqlite"):
        database = url.database
        if database and database != ":memory:":
            p = Path(database)
            if not p.parent.exists():
                p.parent.mkdir(parents=True, exist_ok=True)

    # PostgreSQL: check that psycopg2 is available before SQLAlchemy tries to
    # load the dialect, so the user gets a clear install hint rather than a
    # cryptic "can't load plugin" error.
    if url.drivername.startswith("postgresql"):
        try:
            import psycopg2  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL support requires psycopg2. "
                "Install it with: pip install raggen[postgres]"
            ) from exc

    return create_engine(database_url)
