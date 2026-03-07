from __future__ import annotations

from typing import List, Dict, Any
import json
from sqlalchemy import select
from sqlalchemy.engine import Engine, Connection
from .metadata_schema import documents, chunks, embeddings


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

    def fetch_all_chunk_ids(self, conn: Connection, doc_ids: List[str]) -> List[str]:
        chunk_ids = list(
            conn.scalars(
                select(chunks.c.chunk_id).where(
                    chunks.c.doc_id.in_(doc_ids))
            )
        )
        return chunk_ids

    def fetch_chunks_by_ids(self, conn: Connection, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """
        Fetch chunk rows for the given chunk_ids.

        Returns rows in the same order as `chunk_ids`, not database order.
        Each row contains enough data to build a RetrievedChunk.
        """
        if not chunk_ids:
            return []

        stmt = select(
            chunks.c.chunk_id,
            chunks.c.doc_id,
            chunks.c.text,
            chunks.c.chunk_index,
            chunks.c.start_offset,
            chunks.c.end_offset,
            chunks.c.page_number,
            chunks.c.heading_path_json,
        ).where(chunks.c.chunk_id.in_(chunk_ids))

        rows = conn.execute(stmt).mappings().all()

        # Reorder to match retrieval order
        by_id = {row["chunk_id"]: row for row in rows}

        ordered: list[dict[str, Any]] = []
        for chunk_id in chunk_ids:
            row = by_id.get(chunk_id)
            if row is None:
                continue

            heading_path = None
            raw_heading_path = row.get("heading_path_json")
            if raw_heading_path:
                try:
                    heading_path = json.loads(raw_heading_path)
                except Exception:
                    heading_path = None

            ordered.append(
                {
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "text": row["text"],
                    "chunk_index": row["chunk_index"],
                    "start_offset": row["start_offset"],
                    "end_offset": row["end_offset"],
                    "page_number": row["page_number"],
                    "heading_path": heading_path,
                }
            )

        return ordered

    def delete_documents(self, conn: Connection, doc_ids: List[str]) -> None:
        stmt = documents.delete().where(documents.c.doc_id.in_(doc_ids))
        conn.execute(stmt)


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


def fetch_all_document_ids(engine) -> List[str]:
    with engine.begin() as conn:
        return list(conn.scalars(select(documents.c.doc_id)))
