from __future__ import annotations

from typing import List, Dict, Any, Tuple
from sqlalchemy.engine import Engine
from raggen.core.ingest.config import ProjectConfig
from .vector_backends.base import VectorBackend
from .metadata_store import MetadataStore


def store_document_bundle(
    *,
    engine: Engine,
    cfg: ProjectConfig,
    vector_backend: VectorBackend,
    document_row: Dict[str, Any],
    chunk_rows: List[Dict[str, Any]],
    embeddings: List[Tuple[str, List[float]]],
    embedding_meta_rows: List[Dict[str, Any]],
) -> None:
    """Persist document, chunks, embedding meta, and vectors transactionally."""
    # basic validations
    if any(len(vec) != cfg.embedding.dim for _, vec in embeddings):
        bad = [(cid, len(vec))
               for cid, vec in embeddings if len(vec) != cfg.embedding.dim]
        raise ValueError(f"Embedding dimension mismatch for vectors: {bad}")
    # ensure chunk ids referenced exist among chunk_rows
    chunk_ids = {r["chunk_id"] for r in chunk_rows}
    emb_ids = {cid for cid, _ in embeddings}
    missing = emb_ids - chunk_ids
    if missing:
        raise ValueError(f"Embeddings reference unknown chunk ids: {missing}")

    store = MetadataStore(engine)
    # Use engine.begin() to get transactional Connection
    with engine.begin() as conn:
        store.upsert_document(conn, document_row)
        store.upsert_chunks(conn, chunk_rows)
        store.upsert_embedding_meta(conn, embedding_meta_rows)
        # call vector backend - backends expect Engine or Connection; prefer conn if supported
        # Let backend accept either a Connection or Engine. Prefer passing the connection to avoid nested transactions.
        vector_backend.upsert_vectors(conn, vectors=embeddings, embedding_model_id=cfg.embedding.model_id,
                                      dim=cfg.embedding.dim, normalized=cfg.embedding.normalize)


def delete_documents(
        *,
        engine: Engine,
        vector_backend: VectorBackend,
        documents: List[str],
) -> int:
    """Delete documents and all related MetaData/vectors, returns n docs removed"""
    if not documents:
        return 0

    store = MetadataStore(engine)

    with engine.begin() as conn:
        chunk_ids = store.fetch_all_chunk_ids(conn, documents)
        vector_backend.delete_vectors(engine, chunks=chunk_ids)
        store.delete_documents(conn, documents)
    return len(documents)
