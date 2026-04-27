import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from raggen.core.store.engine import create_engine_from_url


def test_sqlite_creates_missing_parent_directory(tmp_path):
    nested = tmp_path / "a" / "b" / "c" / "rag.db"
    assert not nested.parent.exists()

    create_engine_from_url(f"sqlite:///{nested}")

    assert nested.parent.exists()


def test_sqlite_existing_parent_directory_is_left_alone(tmp_path):
    db_path = tmp_path / "rag.db"
    # tmp_path already exists — should not raise
    engine = create_engine_from_url(f"sqlite:///{db_path}")
    assert engine is not None


def test_sqlite_in_memory_does_not_touch_filesystem(tmp_path):
    engine = create_engine_from_url("sqlite:///:memory:")
    assert engine is not None
    # no files created in tmp_path
    assert list(tmp_path.iterdir()) == []


def test_postgres_missing_psycopg2_raises_clear_error():
    with patch.dict(sys.modules, {"psycopg2": None}):
        with pytest.raises(RuntimeError) as exc_info:
            create_engine_from_url("postgresql://user:pass@localhost/db")

    msg = str(exc_info.value)
    assert "psycopg2" in msg
    assert "raggen[postgres]" in msg
