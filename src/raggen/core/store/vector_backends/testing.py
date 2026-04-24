from __future__ import annotations
from typing import List, Tuple
from sqlalchemy import text, bindparam
from sqlalchemy.engine import Engine, Connection

from .base import VectorBackend


class TestingVectorBackend(VectorBackend):
    key = "testing"

    def supports(self, engine: Engine) -> bool:
        return engine.dialect.name == "sqlite"

    def create_schema(self, engine: Engine, dim: int) -> None:
        with engine.begin() as conn:
            conn.exec_driver_sql("""
                CREATE TABLE IF NOT EXISTS chunk_vectors (
                    chunk_id TEXT PRIMARY KEY,
                    embedding_json TEXT NOT NULL
                )
            """)

    def drop_schema(self, engine: Engine) -> None:
        with engine.begin() as conn:
            conn.exec_driver_sql("DROP TABLE IF EXISTS chunk_vectors")

    def upsert_vectors(
        self,
        conn: Connection,
        *,
        vectors: List[Tuple[str, List[float]]],
        embedding_model_id: str,
        dim: int,
        normalized: bool,
    ) -> None:
        if not vectors:
            return
        for cid, vec in vectors:
            if len(vec) != dim:
                raise ValueError(f"Vector for {cid} has length {len(vec)} != {dim}")

        def vec_to_str(v: List[float]) -> str:
            return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"

        stmt = text("""
            INSERT OR REPLACE INTO chunk_vectors(chunk_id, embedding_json)
            VALUES (:chunk_id, :embedding_json)
        """)
        conn.execute(
            stmt,
            [
                {"chunk_id": cid, "embedding_json": vec_to_str(vec)}
                for cid, vec in vectors
            ],
        )

    def delete_vectors(self, conn: Connection, *, chunks: List[str]) -> None:
        if not chunks:
            return
        stmt = text("""
            DELETE FROM chunk_vectors
            WHERE chunk_id IN :chunk_ids
        """).bindparams(bindparam("chunk_ids", expanding=True))
        conn.execute(stmt, {"chunk_ids": chunks})
