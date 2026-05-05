"""
Contract tests for real vector backend implementations.

SQLiteVec tests always run (sqlite-vec is a core dependency).
PgVector tests are skipped unless POSTGRES_URL is set in the environment.

To run pgvector tests locally:
    POSTGRES_URL=postgresql://user:pass@localhost:5432/db pytest tests/test_vector_backend_contracts.py
"""
from __future__ import annotations

import pytest


DIM = 4
MODEL_ID = "test-model"


def _unit_vec(n: int, dim: int) -> list[float]:
    """Return a simple deterministic unit vector for test data."""
    v = [0.0] * dim
    v[n % dim] = 1.0
    return v


# ---------------------------------------------------------------------------
# Backend + engine parametrization
# ---------------------------------------------------------------------------


@pytest.fixture(params=["sqlite_vec", "pgvector"])
def backend_and_engine(request, sqlite_engine):
    """Yields (backend_instance, engine) for each available vector backend.

    The pgvector variant is skipped unless POSTGRES_URL is set. The pg_engine
    fixture is NOT requested here to avoid skipping the sqlite_vec variant
    as a side effect.
    """
    if request.param == "sqlite_vec":
        from raggen.core.store.vector_backends.sqlite_vec import SQLiteVecBackend
        return SQLiteVecBackend(), sqlite_engine
    else:
        import os
        url = os.environ.get("POSTGRES_URL")
        if not url:
            pytest.skip("POSTGRES_URL not set")
        pytest.importorskip("psycopg2")
        from sqlalchemy import create_engine
        from raggen.core.store.vector_backends.pgvector import PgVectorBackend
        return PgVectorBackend(), create_engine(url)


@pytest.fixture(autouse=True)
def _clean_schema(backend_and_engine):
    """Drop schema before and after each test to guarantee a clean slate."""
    backend, engine = backend_and_engine
    backend.drop_schema(engine)
    yield
    backend.drop_schema(engine)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def test_create_schema_is_idempotent(backend_and_engine):
    backend, engine = backend_and_engine
    backend.create_schema(engine, dim=DIM)
    backend.create_schema(engine, dim=DIM)  # second call must not raise


def test_drop_schema_is_safe_when_absent(backend_and_engine):
    backend, engine = backend_and_engine
    backend.drop_schema(engine)  # no schema exists yet — must not raise


def test_upsert_and_search_roundtrip(backend_and_engine):
    backend, engine = backend_and_engine
    backend.create_schema(engine, dim=DIM)

    vec = _unit_vec(0, DIM)
    with engine.begin() as conn:
        backend.upsert_vectors(
            conn,
            vectors=[("chunk-1", vec)],
            embedding_model_id=MODEL_ID,
            dim=DIM,
            normalized=True,
        )

    results = backend.search(engine, query_vector=vec, top_k=1)

    assert len(results) == 1
    chunk_id, score = results[0]
    assert chunk_id == "chunk-1"
    assert isinstance(score, float)


def test_search_returns_ascending_scores(backend_and_engine):
    backend, engine = backend_and_engine
    backend.create_schema(engine, dim=DIM)

    # Insert 3 vectors. The query is identical to vec_0, so it should be closest.
    vecs = [("c0", _unit_vec(0, DIM)), ("c1", _unit_vec(1, DIM)), ("c2", _unit_vec(2, DIM))]
    with engine.begin() as conn:
        backend.upsert_vectors(
            conn,
            vectors=vecs,
            embedding_model_id=MODEL_ID,
            dim=DIM,
            normalized=True,
        )

    results = backend.search(engine, query_vector=_unit_vec(0, DIM), top_k=3)

    assert len(results) == 3
    scores = [score for _, score in results]
    assert scores == sorted(scores), "scores must be in ascending order (lower = closer)"
    assert results[0][0] == "c0", "exact match must be top result"


def test_search_respects_top_k(backend_and_engine):
    backend, engine = backend_and_engine
    backend.create_schema(engine, dim=DIM)

    vecs = [(f"c{i}", _unit_vec(i, DIM)) for i in range(4)]
    with engine.begin() as conn:
        backend.upsert_vectors(
            conn,
            vectors=vecs,
            embedding_model_id=MODEL_ID,
            dim=DIM,
            normalized=True,
        )

    results = backend.search(engine, query_vector=_unit_vec(0, DIM), top_k=2)
    assert len(results) <= 2


def test_upsert_overwrites_existing_vector(backend_and_engine):
    backend, engine = backend_and_engine
    backend.create_schema(engine, dim=DIM)

    original = _unit_vec(0, DIM)
    replacement = _unit_vec(1, DIM)

    with engine.begin() as conn:
        backend.upsert_vectors(
            conn,
            vectors=[("c0", original)],
            embedding_model_id=MODEL_ID,
            dim=DIM,
            normalized=True,
        )
    with engine.begin() as conn:
        backend.upsert_vectors(
            conn,
            vectors=[("c0", replacement)],
            embedding_model_id=MODEL_ID,
            dim=DIM,
            normalized=True,
        )

    # Querying with the replacement vector should hit c0 as top result.
    results = backend.search(engine, query_vector=replacement, top_k=1)
    assert results[0][0] == "c0"


def test_delete_vectors_removes_entries(backend_and_engine):
    backend, engine = backend_and_engine
    backend.create_schema(engine, dim=DIM)

    vecs = [("keep", _unit_vec(0, DIM)), ("remove", _unit_vec(1, DIM))]
    with engine.begin() as conn:
        backend.upsert_vectors(
            conn,
            vectors=vecs,
            embedding_model_id=MODEL_ID,
            dim=DIM,
            normalized=True,
        )
    with engine.begin() as conn:
        backend.delete_vectors(conn, chunks=["remove"])

    results = backend.search(engine, query_vector=_unit_vec(1, DIM), top_k=10)
    chunk_ids = [cid for cid, _ in results]
    assert "remove" not in chunk_ids
    assert "keep" in chunk_ids


def test_delete_vectors_empty_list_is_safe(backend_and_engine):
    backend, engine = backend_and_engine
    backend.create_schema(engine, dim=DIM)

    with engine.begin() as conn:
        backend.delete_vectors(conn, chunks=[])  # must not raise
