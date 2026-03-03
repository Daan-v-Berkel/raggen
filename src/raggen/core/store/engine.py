from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from pathlib import Path


def create_engine_from_url(database_url: str) -> Engine:
    # For sqlite file URLs, ensure parent directory exists
    url = make_url(database_url)
    if url.drivername.startswith("sqlite"):
        # file path form: sqlite:////abs/path or sqlite:///relative/path
        # extract database file path if present (empty for in-memory)
        database = url.database
        if database and database != ":memory":
            p = Path(database)
            parent = p.parent
            if not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(database_url)

    # Backwards-compatibility: some code (and tests) execute raw SQL strings via
    # Connection.execute("SELECT ..."), which is not accepted by SQLAlchemy 2.x.
    # Wrap engine.connect() to return a proxy connection that accepts string SQL
    # by converting it to sqlalchemy.text() before delegating.
    from sqlalchemy import text

    class _ConnProxy:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, statement, *args, **kwargs):
            if isinstance(statement, str):
                return self._conn.execute(text(statement), *args, **kwargs)
            return self._conn.execute(statement, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def __enter__(self):
            self._ctx = self._conn.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._conn.__exit__(exc_type, exc, tb)

    # store original connect and replace with wrapper
    _orig_connect = engine.connect

    def _connect_wrapper(*args, **kwargs):
        conn = _orig_connect(*args, **kwargs)
        return _ConnProxy(conn)

    engine.connect = _connect_wrapper
    return engine
