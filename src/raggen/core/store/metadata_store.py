from __future__ import annotations

from typing import List, Dict, Any
from sqlalchemy.engine import Engine, Connection
from .metadata_schema import documents, chunks, embeddings
from sqlalchemy import insert, text


class MetadataStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def upsert_document(self, conn: Connection, row: Dict[str, Any]) -> None:
        upsert_rows(conn, documents, [row], ["doc_id"]) 

    def upsert_chunks(self, conn: Connection, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        upsert_rows(conn, chunks, rows, ["chunk_id"]) 

    def upsert_embedding_meta(self, conn: Connection, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        upsert_rows(conn, embeddings, rows, ["chunk_id"]) 


def upsert_rows(conn: Connection, table, rows: List[Dict[str, Any]], pk_cols: List[str]) -> None:
    """Portable upsert: delete existing PKs then insert all rows in same transaction."""
    if not rows:
        return
    # collect pk values
    pk_vals = [tuple(row[col] for col in pk_cols) for row in rows]
    # if single PK column, do simple IN
    if len(pk_cols) == 1:
        pk = pk_cols[0]
        vals = [v[0] for v in pk_vals]
        # delete existing
        del_stmt = table.delete().where(table.c[pk].in_(vals))
        conn.execute(del_stmt)
    else:
        # composite keys: delete per row
        for vals in pk_vals:
            cond = None
            for col, val in zip(pk_cols, vals):
                c = table.c[col] == val
                cond = c if cond is None else (cond & c)
            conn.execute(table.delete().where(cond))
    # bulk insert
    conn.execute(table.insert(), rows)
