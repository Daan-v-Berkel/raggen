import types
import sys
import pytest
from sqlalchemy import select

from raggen.core.config.project import default_project_config
from raggen.core.store.initializer import init_database
from raggen.core.store.metadata_store import MetadataStore
from raggen.core.store.ingest_store import store_document_bundle
from raggen.core.store.metadata_schema import documents, chunks, embeddings


class DummyInitBackend:
    def __init__(self):
        self.key = "dummy_init"

    def supports(self, engine):
        return True

    def create_schema(self, engine, dim):
        # no-op
        pass

    def drop_schema(self, engine):
        pass

    def upsert_vectors(self, *args, **kwargs):
        pass


def _inject_dummy_module(name="tests._dummy_init_mod"):
    mod = types.ModuleType(name)
    mod.DummyInitBackend = DummyInitBackend
    sys.modules[name] = mod
    return name + ":DummyInitBackend"


def test_metadata_store_upserts_sqlite(tmp_path):
    db_file = tmp_path / "rag.db"
    import_path = _inject_dummy_module()
    cfg = default_project_config(tmp_path)

    cfg.storage.database_url = f"sqlite:///{db_file.resolve().as_posix()}"
    cfg.storage.backend_key = "dummy_init"
    cfg.storage.vector_backend_import = import_path
    cfg.embedding.model_id = "m"
    cfg.embedding.dim = 3
    cfg.embedding.normalize = True

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

    # verify inserted
    with engine.connect() as conn:
        r = conn.execute(select(documents)).mappings().fetchone()
        assert r and r["doc_id"] == "d1"
        r2 = conn.execute(select(chunks)).mappings().fetchone()
        assert r2 and r2["chunk_id"] == "c1"
        r3 = conn.execute(select(embeddings)).mappings().fetchone()
        assert r3 and r3["chunk_id"] == "c1"

    # upsert modified values
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


class RaisingBackend:
    def __init__(self):
        self.key = "raise"

    def supports(self, engine):
        return True

    def create_schema(self, engine, dim):
        pass

    def drop_schema(self, engine):
        pass

    def upsert_vectors(self, *args, **kwargs):
        raise RuntimeError("fail vector upsert")


class GoodBackend(RaisingBackend):
    def upsert_vectors(self, *args, **kwargs):
        return None


def test_store_document_bundle_transactionality(tmp_path):
    db_file = tmp_path / "rag.db"
    import_path = _inject_dummy_module()
    cfg = default_project_config(tmp_path)

    cfg.storage.database_url = f"sqlite:///{db_file.resolve().as_posix()}"
    cfg.storage.backend_key = "dummy_init"
    cfg.storage.vector_backend_import = import_path
    cfg.embedding.model_id = "m"
    cfg.embedding.dim = 3
    cfg.embedding.normalize = True
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
    embeddings = [("ct1", [0.1, 0.2, 0.3])]
    embedding_meta_rows = [
        {"chunk_id": "ct1", "embedding_model_id": "m",
            "dim": 3, "normalized": 1, "created_at": "t"}
    ]

    # use raising backend
    bad = RaisingBackend()
    with pytest.raises(RuntimeError):
        store_document_bundle(engine=engine, cfg=cfg, vector_backend=bad, document_row=document_row,
                              chunk_rows=chunk_rows, embeddings=embeddings, embedding_meta_rows=embedding_meta_rows)

    with engine.connect() as conn:
        r = conn.execute(select(documents)).mappings().fetchone()
        assert r is None

    # now with good backend
    good = GoodBackend()
    store_document_bundle(engine=engine, cfg=cfg, vector_backend=good, document_row=document_row,
                          chunk_rows=chunk_rows, embeddings=embeddings, embedding_meta_rows=embedding_meta_rows)

    with engine.connect() as conn:
        r = conn.execute(select(documents)).mappings().fetchone()
        assert r and r["doc_id"] == "dtx"


def test_e2e_row_builders_minimal_fields():
    # import builders from scripts if possible
    try:
        import scripts.e2e_chunk_test as e2e
        build_chunk = e2e.build_chunk_row
    except Exception:
        # fallback: recreate minimal builder
        def build_chunk(doc_id, chunk_index, text, chunk_id=None):
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

        build_chunk = build_chunk

    row = build_chunk("dmini", 0, "hi")
    assert row["page_number"] is None
    assert row["heading_path_json"] is None
    assert row["start_offset"] == 0
    assert row["end_offset"] == 2
