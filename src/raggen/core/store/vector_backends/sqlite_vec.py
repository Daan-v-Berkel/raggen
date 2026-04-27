from __future__ import annotations

import struct
from typing import List, Tuple

from sqlalchemy import text, bindparam
from sqlalchemy.engine import Engine, Connection

from raggen.core.store.vector_backends.base import VectorBackend


def _serialize_f32(vec: List[float]) -> bytes:
    """Serialize floats into sqlite-vec expected raw float32 bytes."""
    # sqlite-vec expects float32 bytes; struct 'f' is C float (usually 32-bit)
    return struct.pack(f"{len(vec)}f", *[float(x) for x in vec])


class SQLiteVecBackend(VectorBackend):
    key = "sqlite_vec"

    # ---- sqlite-vec loading ----

    def _ensure_vec_loaded(self, conn: Connection) -> None:
        """
        Ensure sqlite-vec is loaded for the *current sqlite3 connection*.

        Uses the official python package if installed. This is the recommended path.
        """
        # Get the raw sqlite3 connection from SQLAlchemy
        raw = getattr(conn.connection, "driver_connection", None) or conn.connection

        try:
            import sqlite_vec  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "sqlite-vec backend selected but the 'sqlite-vec' package is not installed. "
                "Install it with: pip install sqlite-vec"
            ) from e

        # Load extension into this raw connection
        try:
            raw.enable_load_extension(True)
            sqlite_vec.load(raw)
        finally:
            # lock it back down
            try:
                raw.enable_load_extension(False)
            except Exception:
                pass

        # Sanity check: vec_version() should exist now
        conn.exec_driver_sql("SELECT vec_version()").scalar_one()

    def supports(self, engine: Engine) -> bool:
        return engine.dialect.name == "sqlite"

    # ---- schema ----

    def create_schema(self, engine: Engine, dim: int) -> None:
        """
        Creates:
          - chunk_vectors_map: maps chunk_id (TEXT) -> id (INTEGER)
          - chunk_vectors: sqlite-vec virtual table storing embedding at rowid = id
        """
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")

        with engine.begin() as conn:
            self._ensure_vec_loaded(conn)

            # Mapping table (normal SQLite table)
            conn.exec_driver_sql("""
                CREATE TABLE IF NOT EXISTS chunk_vectors_map (
                    id INTEGER PRIMARY KEY,
                    chunk_id TEXT NOT NULL UNIQUE
                )
                """)

            # Vec virtual table (only embedding; rowid used as key)
            conn.exec_driver_sql(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors
                USING vec0(embedding float[{dim}])
                """)

    def drop_schema(self, engine: Engine) -> None:
        with engine.begin() as conn:
            # drop vec virtual table first, then mapping
            try:
                conn.exec_driver_sql("DROP TABLE IF EXISTS chunk_vectors")
            except Exception:
                pass
            conn.exec_driver_sql("DROP TABLE IF EXISTS chunk_vectors_map")

    # ---- upsert ----

    def upsert_vectors(
        self,
        conn: Connection,
        *,
        vectors: List[Tuple[str, List[float]]],
        embedding_model_id: str,  # not used by sqlite-vec table; metadata lives elsewhere
        dim: int,
        normalized: bool,  # not used here; metadata lives elsewhere
    ) -> None:
        if not vectors:
            return

        for cid, vec in vectors:
            if len(vec) != dim:
                raise ValueError(f"Vector for {cid} has length {len(vec)} != {dim}")

        self._ensure_vec_loaded(conn)

        # Mapping table: safe to use ON CONFLICT (this is a normal table)
        insert_map = text("""
            INSERT INTO chunk_vectors_map (chunk_id)
            VALUES (:chunk_id)
            ON CONFLICT(chunk_id) DO NOTHING
        """)

        select_id = text("""
            SELECT id
            FROM chunk_vectors_map
            WHERE chunk_id = :chunk_id
        """)

        # Virtual table: must do manual "upsert"
        update_vec = text("""
            UPDATE chunk_vectors
            SET embedding = :embedding
            WHERE rowid = :rowid
        """)

        insert_vec = text("""
            INSERT INTO chunk_vectors(rowid, embedding)
            VALUES (:rowid, :embedding)
        """)

        for chunk_id, vec in vectors:
            conn.execute(insert_map, {"chunk_id": chunk_id})
            rowid = conn.execute(select_id, {"chunk_id": chunk_id}).scalar_one()
            params = {"rowid": int(rowid), "embedding": _serialize_f32(vec)}

            res = conn.execute(update_vec, params)
            if res.rowcount == 0:
                conn.execute(insert_vec, params)

    # ---- delete ----

    def delete_vectors(
        self,
        conn: Connection,
        *,
        chunks: List[str],
    ) -> None:
        if not chunks:
            return

        self._ensure_vec_loaded(conn)

        ids_stmt = text("""
            SELECT id
            FROM chunk_vectors_map
            WHERE chunk_id IN :chunk_ids
            """).bindparams(bindparam("chunk_ids", expanding=True))

        rowids = list(conn.scalars(ids_stmt, {"chunk_ids": chunks}))
        if rowids:
            del_vec = text(
                "DELETE FROM chunk_vectors WHERE rowid IN :rowids"
            ).bindparams(bindparam("rowids", expanding=True))
            conn.execute(del_vec, {"rowids": rowids})

        del_map = text(
            "DELETE FROM chunk_vectors_map WHERE chunk_id IN :chunk_ids"
        ).bindparams(bindparam("chunk_ids", expanding=True))
        conn.execute(del_map, {"chunk_ids": chunks})

    def search(
        self,
        engine,
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
                m.chunk_id,
                v.distance
            FROM chunk_vectors AS v
            JOIN chunk_vectors_map AS m
              ON m.id = v.rowid
            WHERE v.embedding MATCH :query_embedding
              AND k = :top_k
            ORDER BY v.distance ASC
        """)

        with engine.connect() as conn:
            self._ensure_vec_loaded(conn)
            rows = conn.execute(
                stmt,
                {
                    "query_embedding": _serialize_f32(query_vector),
                    "top_k": top_k,
                },
            ).fetchall()

        return [(row[0], float(row[1])) for row in rows]
