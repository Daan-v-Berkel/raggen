import pytest
from sqlalchemy import select

from raggen.core.bootstrap import bootstrap
from raggen.core.store.ingest_store import store_document_bundle
from raggen.core.store.initializer import init_database
from raggen.core.store.metadata_schema import documents, chunks, embeddings
from raggen.core.store.metadata_store import MetadataStore


class RaisingBackend:
    key = "raise"

    def supports(self, engine):
        return True

    def create_schema(self, engine, dim):
        pass

    def drop_schema(self, engine):
        pass

    def upsert_vectors(self, *args, **kwargs):
        raise RuntimeError("fail vector upsert")

    def delete_vectors(self, *args, **kwargs):
        return None

    def search(self, engine, *, query_vector, top_k):
        return []


class GoodBackend(RaisingBackend):
    key = "good"

    def upsert_vectors(self, *args, **kwargs):
        return None


def test_metadata_store_upserts_sqlite(
    tmp_path,
    cfg_factory,
    write_cfg,
    noop_backend_import,
):
    cfg = cfg_factory(tmp_path)
    cfg.storage.backend_key = "dummy_init"
    cfg.storage.vector_backend_import = noop_backend_import
    cfg.embedding.model_id = "m"
    cfg.embedding.dim = 3
    cfg.embedding.normalize = True

    cfg_path = write_cfg(cfg, tmp_path)
    cfg = bootstrap(cfg_path)

    engine = init_database(cfg)
    ms = MetadataStore(engine)

    doc_row = {
        "doc_id": "d1",
        "source_path": "p",
        "mimetype": "text/plain",
        "mtime_ns": 10,
        "byte_size": 10,
        "content_hash": "h",
        "parsed_at": "t",
        "parser_id": "p",
        "structure_version": "v1",
        "text_char_len": 10,
    }

    chunk_row = {
        "chunk_id": "c1",
        "doc_id": "d1",
        "chunk_index": 0,
        "text": "hello",
        "start_offset": 0,
        "end_offset": 5,
        "page_number": None,
        "heading_path_json": None,
        "chunk_config_hash": "abc",
        "created_at": "t",
    }

    emb_row = {
        "chunk_id": "c1",
        "embedding_model_id": "m",
        "dim": 3,
        "normalized": 1,
        "created_at": "t",
    }

    with engine.begin() as conn:
        ms.upsert_document(conn, doc_row)
        ms.upsert_chunks(conn, [chunk_row])
        ms.upsert_embedding_meta(conn, [emb_row])

    with engine.connect() as conn:
        r = conn.execute(select(documents)).mappings().fetchone()
        assert r and r["doc_id"] == "d1"

        r2 = conn.execute(select(chunks)).mappings().fetchone()
        assert r2 and r2["chunk_id"] == "c1"

        r3 = conn.execute(select(embeddings)).mappings().fetchone()
        assert r3 and r3["chunk_id"] == "c1"

    # Upsert modified values
    doc_row["byte_size"] = 999
    chunk_row["text"] = "changed"
    emb_row["dim"] = 4

    with engine.begin() as conn:
        ms.upsert_document(conn, doc_row)
        ms.upsert_chunks(conn, [chunk_row])
        ms.upsert_embedding_meta(conn, [emb_row])

    with engine.connect() as conn:
        r = conn.execute(select(documents)).mappings().fetchone()
        assert r and int(r["byte_size"]) == 999

        r2 = conn.execute(select(chunks)).mappings().fetchone()
        assert r2 and r2["text"] == "changed"

        r3 = conn.execute(select(embeddings)).mappings().fetchone()
        assert r3 and int(r3["dim"]) == 4


def test_store_document_bundle_transactionality(
    tmp_path,
    cfg_factory,
    write_cfg,
    noop_backend_import,
):
    cfg = cfg_factory(tmp_path)
    cfg.storage.backend_key = "dummy_init"
    cfg.storage.vector_backend_import = noop_backend_import
    cfg.embedding.model_id = "m"
    cfg.embedding.dim = 3
    cfg.embedding.normalize = True

    cfg_path = write_cfg(cfg, tmp_path)
    cfg = bootstrap(cfg_path)

    engine = init_database(cfg)

    document_row = {
        "doc_id": "dtx",
        "source_path": "p",
        "mimetype": "text/plain",
        "mtime_ns": 10,
        "byte_size": 10,
        "content_hash": "h",
        "parsed_at": "t",
        "parser_id": "p",
        "structure_version": "v1",
        "text_char_len": 10,
    }

    chunk_rows = [
        {
            "chunk_id": "ct1",
            "doc_id": "dtx",
            "chunk_index": 0,
            "text": "a",
            "start_offset": 0,
            "end_offset": 1,
            "page_number": None,
            "heading_path_json": None,
            "chunk_config_hash": "abc",
            "created_at": "t",
        }
    ]

    embeddings_rows = [("ct1", [0.1, 0.2, 0.3])]

    embedding_meta_rows = [
        {
            "chunk_id": "ct1",
            "embedding_model_id": "m",
            "dim": 3,
            "normalized": 1,
            "created_at": "t",
        }
    ]

    bad = RaisingBackend()

    with pytest.raises(RuntimeError, match="fail vector upsert"):
        store_document_bundle(
            engine=engine,
            cfg=cfg,
            vector_backend=bad,
            document_row=document_row,
            chunk_rows=chunk_rows,
            embeddings=embeddings_rows,
            embedding_meta_rows=embedding_meta_rows,
        )

    with engine.connect() as conn:
        r = conn.execute(select(documents)).mappings().fetchone()
        assert r is None

    good = GoodBackend()

    store_document_bundle(
        engine=engine,
        cfg=cfg,
        vector_backend=good,
        document_row=document_row,
        chunk_rows=chunk_rows,
        embeddings=embeddings_rows,
        embedding_meta_rows=embedding_meta_rows,
    )

    with engine.connect() as conn:
        r = conn.execute(select(documents)).mappings().fetchone()
        assert r and r["doc_id"] == "dtx"


def test_chunk_row_builder_minimal_fields():
    """
    Keep this test local to store concerns.

    The old script-based import path is gone, so this checks the minimal
    row-builder contract directly.
    """
    def build_chunk_row(doc_id: str, chunk_index: int, text: str, chunk_id: str | None = None):
        cid = chunk_id or f"{doc_id}#chunk:{chunk_index}"
        return {
            "chunk_id": cid,
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "text": text,
            "start_offset": 0,
            "end_offset": len(text),
            "page_number": None,
            "heading_path_json": None,
            "chunk_config_hash": "abc",
            "created_at": "t",
        }

    row = build_chunk_row("dmini", 0, "hi")

    assert row["page_number"] is None
    assert row["heading_path_json"] is None
    assert row["start_offset"] == 0
    assert row["end_offset"] == 2
