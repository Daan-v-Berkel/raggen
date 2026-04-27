from __future__ import annotations

from typing import List, Tuple
from sqlalchemy import text, bindparam
from sqlalchemy.engine import Engine, Connection
from .base import VectorBackend


def _pgvector_literal(vector: List[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


class PgVectorBackend(VectorBackend):
    key = "pgvector"

    def supports(self, engine: Engine) -> bool:
        return engine.dialect.name == "postgresql"

    def create_schema(self, engine: Engine, dim: int) -> None:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS chunk_vectors (
                    chunk_id TEXT PRIMARY KEY,
                    embedding vector({dim}) NOT NULL
                )
            """))

    def drop_schema(self, engine: Engine) -> None:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS chunk_vectors"))

    def upsert_vectors(
        self,
        conn: Connection,
        *,
        vectors: List[Tuple[str, List[float]]],
        embedding_model_id: str,
        dim: int,
        normalized: bool,
    ) -> None:
        for cid, vec in vectors:
            if len(vec) != dim:
                raise ValueError(f"Vector for {cid} has length {len(vec)} != {dim}")

        stmt = text("""
            INSERT INTO chunk_vectors (chunk_id, embedding)
            VALUES (:chunk_id, CAST(:vec AS vector))
            ON CONFLICT (chunk_id) DO UPDATE
                SET embedding = EXCLUDED.embedding
        """)

        for cid, vec in vectors:
            conn.execute(stmt, {
                "chunk_id": cid,
                "vec": _pgvector_literal(vec),
            })

    def delete_vectors(
        self,
        conn: Connection,
        *,
        chunks: List[str],
    ) -> None:
        if not chunks:
            return

        stmt = text("""
            DELETE FROM chunk_vectors
            WHERE chunk_id IN :chunk_ids
        """).bindparams(bindparam("chunk_ids", expanding=True))
        conn.execute(stmt, {"chunk_ids": chunks})

    def search(
        self,
        engine: Engine,
        *,
        query_vector: list[float],
        top_k: int,
    ) -> list[tuple[str, float]]:
        if not query_vector:
            raise ValueError("query_vector must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        stmt = text("""
            SELECT
                chunk_id,
                embedding <=> CAST(:query_embedding AS vector) AS distance
            FROM chunk_vectors
            ORDER BY embedding <=> CAST(:query_embedding AS vector) ASC
            LIMIT :top_k
        """)

        with engine.connect() as conn:
            rows = conn.execute(
                stmt,
                {
                    "query_embedding": _pgvector_literal(query_vector),
                    "top_k": top_k,
                },
            ).fetchall()

        return [(row[0], float(row[1])) for row in rows]
