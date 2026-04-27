from __future__ import annotations

from sqlalchemy import (
    MetaData,
    Table,
    Column,
    Boolean,
    Integer,
    Text,
    text,
    ForeignKey,
    Index,
    CheckConstraint,
)

metadata = MetaData()

rag_project = Table(
    "rag_project",
    metadata,
    Column("id", Integer, primary_key=True),
    Column(
        "created_at", Text, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    ),
    Column("schema_version", Text, nullable=False),
    Column("backend_key", Text, nullable=False),
    Column("database_url", Text, nullable=False),
    Column("embedding_model_id", Text, nullable=False),
    Column("embedding_dim", Integer, nullable=False),
    Column("embedding_normalized", Boolean, nullable=False),
    Column("query_model_id", Text, nullable=True),
    # Column("chunk_config_hash", Text, nullable=False),
    Column("notes_json", Text, nullable=True),
    CheckConstraint("id = 1", name="rag_project_singleton_chk"),
)

documents = Table(
    "documents",
    metadata,
    Column("doc_id", Text, primary_key=True),
    Column("source_path", Text, nullable=False),
    Column("mimetype", Text, nullable=False),
    Column("byte_size", Integer, nullable=False),
    Column("content_hash", Text, nullable=False),
    Column("parsed_at", Text, nullable=False),
    Column("mtime_ns", Integer, nullable=False),
    Column("parser_id", Text, nullable=False),
    Column("structure_version", Text, nullable=False),
    Column("text_char_len", Integer, nullable=False),
)

chunks = Table(
    "chunks",
    metadata,
    Column("chunk_id", Text, primary_key=True),
    Column(
        "doc_id",
        Text,
        ForeignKey("documents.doc_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("chunk_index", Integer, nullable=False),
    Column("text", Text, nullable=False),
    Column("start_offset", Integer, nullable=False),
    Column("end_offset", Integer, nullable=False),
    Column("page_number", Integer, nullable=True),
    Column("heading_path_json", Text, nullable=True),
    Column("chunk_config_hash", Text, nullable=False),
    Column("created_at", Text, nullable=False),
)
Index("chunks_doc_id_idx", chunks.c.doc_id)
Index("chunks_doc_order_idx", chunks.c.doc_id, chunks.c.chunk_index)

embeddings = Table(
    "embeddings",
    metadata,
    Column(
        "chunk_id",
        Text,
        ForeignKey("chunks.chunk_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("embedding_model_id", Text, nullable=False),
    Column("dim", Integer, nullable=False),
    Column("normalized", Boolean, nullable=False),
    Column("created_at", Text, nullable=False),
)
