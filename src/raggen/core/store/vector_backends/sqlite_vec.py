from __future__ import annotations

from typing import List, Tuple
from sqlalchemy.engine import Engine
from .base import VectorBackend
import os


class SQLiteVecBackend(VectorBackend):
    key = "sqlite_vec"

    def _load_vec_extension(self, conn):
        # Attempt to load extension if path provided
        path = os.environ.get("RAGGEN_SQLITE_VEC_PATH")
        try:
            if path:
                conn.execute(f"SELECT load_extension('{path}')")
            else:
                # best-effort: try common names; some systems allow load_extension without path
                conn.execute("SELECT load_extension('vec')")
        except Exception:
            # ignore here; supports() will handle capability checks
            pass

    def supports(self, engine: Engine) -> bool:
        # Only check dialect; availability of sqlite-vec extension is optional. We provide a fallback.
        return engine.dialect.name == "sqlite"

    def create_schema(self, engine: Engine, dim: int) -> None:
        # Try to create a vec virtual table; if that fails (extension not available), create a fallback table.
        virtual_ddl = f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec(embedding, dims={dim});
        """
        fallback_ddl = f"""
        CREATE TABLE IF NOT EXISTS chunk_vectors_flat (
            chunk_id TEXT PRIMARY KEY,
            embedding_json TEXT NOT NULL,
            embedding_model_id TEXT NOT NULL,
            normalized BOOLEAN NOT NULL,
            created_at DATETIME DEFAULT (CURRENT_TIMESTAMP)
        );
        """
        with engine.begin() as conn:
            try:
                conn.execute(virtual_ddl)
            except Exception:
                # fallback table if vec extension unavailable
                conn.execute(fallback_ddl)

    def drop_schema(self, engine: Engine) -> None:
        with engine.begin() as conn:
            # attempt both drops; virtual table drop may fail if not present
            try:
                conn.execute("DROP TABLE IF EXISTS chunk_vectors")
            except Exception:
                pass
            try:
                conn.execute("DROP TABLE IF EXISTS chunk_vectors_flat")
            except Exception:
                pass

    def upsert_vectors(
        self,
        engine: Engine,
        *,
        vectors: List[Tuple[str, List[float]]],
        embedding_model_id: str,
        dim: int,
        normalized: bool,
    ) -> None:
        # validate dims
        for cid, vec in vectors:
            if len(vec) != dim:
                raise ValueError(f"Vector for {cid} has length {len(vec)} != {dim}")
        # Use fallback table for portability: store JSON array
        with engine.begin() as conn:
            for cid, vec in vectors:
                vec_str = "[" + ",".join(str(float(x)) for x in vec) + "]"
                conn.execute(
                    "INSERT OR REPLACE INTO chunk_vectors_flat (chunk_id, embedding_json, embedding_model_id, normalized) VALUES (:cid, :vec, :model, :norm)",
                    {"cid": cid, "vec": vec_str, "model": embedding_model_id, "norm": normalized},
                )
