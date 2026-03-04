from __future__ import annotations

from typing import List, Tuple
from sqlalchemy import text, bindparam
from sqlalchemy.engine import Engine
from .base import VectorBackend
from .base import List as _List  # type: ignore


class PgVectorBackend(VectorBackend):
    key = "pgvector"

    def supports(self, engine: Engine) -> bool:
        return engine.dialect.name in ("postgresql", "postgres")

    def create_schema(self, engine: Engine, dim: int) -> None:
        # Create extension and table
        ddl = f"""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS chunk_vectors (
            chunk_id TEXT PRIMARY KEY,
            embedding vector({dim}) NOT NULL,
            embedding_model_id TEXT NOT NULL,
            normalized BOOLEAN NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
        with engine.begin() as conn:
            conn.execute(ddl)

    def drop_schema(self, engine: Engine) -> None:
        with engine.begin() as conn:
            conn.execute("DROP TABLE IF EXISTS chunk_vectors")

    def upsert_vectors(
        self,
        engine_or_conn,
        *,
        vectors: List[Tuple[str, List[float]]],
        embedding_model_id: str,
        dim: int,
        normalized: bool,
    ) -> None:
        # validate dims
        for cid, vec in vectors:
            if len(vec) != dim:
                raise ValueError(
                    f"Vector for {cid} has length {len(vec)} != {dim}")
        # Insert using parameterized query; use string form for vector literal

        def _do_upsert(conn):
            stmt = (
                "INSERT INTO chunk_vectors (chunk_id, embedding, embedding_model_id, normalized) VALUES (:chunk_id, :vec::vector(%d), :model, :norm) "
                "ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding, embedding_model_id = EXCLUDED.embedding_model_id, normalized = EXCLUDED.normalized"
                % dim
            )
            for cid, vec in vectors:
                vec_str = "[" + ",".join(str(float(x)) for x in vec) + "]"
                conn.execute(stmt, {"chunk_id": cid, "vec": vec_str,
                             "model": embedding_model_id, "norm": normalized})

        from sqlalchemy.engine import Connection, Engine
        if isinstance(engine_or_conn, Connection):
            _do_upsert(engine_or_conn)
        elif isinstance(engine_or_conn, Engine):
            with engine_or_conn.begin() as conn:
                _do_upsert(conn)
        else:
            _do_upsert(engine_or_conn)

    def delete_vectors(
        self,
        engine,
        *,
        chunks: List[str],
    ) -> None:
        if not chunks:
            return

        stmt = text("""
            DELETE FROM chunk_vectors
            WHERE chunk_id IN :chunk_ids
        """).bindparams(
            bindparam("chunk_ids", expanding=True)
        )

        with engine.begin() as conn:
            conn.execute(stmt, {"chunk_ids": chunks})
