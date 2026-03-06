from types import SimpleNamespace
import numpy as np

from raggen.core.ingest.ingest_service import init_and_ingest
from raggen.core.ingest.config import default_project_config
from raggen.core.parsing.parser import ParserService
from raggen.core.chunking.chunker import Chunker


def _dummy_embedder_factory(*args, **kwargs):
    class Dummy:
        def __init__(self, model_id=None):
            self.model_id = model_id

        def dim(self):
            return 4

        def embed_texts(self, texts, batch_size=32, normalize=True):
            return np.zeros((len(texts), 4), dtype=np.float32)
    return Dummy()


def test_empty_raw_file_is_skipped(tmp_path, monkeypatch):
    root = tmp_path
    f = root / "empty.txt"
    f.write_bytes(b"")

    cfg = default_project_config(root)
    cfg.storage.database_url = f"sqlite:///{(root / '.rag' / 'rag.db').resolve().as_posix()}"

    # Prevent heavy model loading
    monkeypatch.setattr("raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
                        lambda model_id: _dummy_embedder_factory())

    # Ensure parser is NOT called
    def _fail_parse(self, inp):
        raise AssertionError("parser should not be called for empty raw files")
    monkeypatch.setattr(ParserService, "parse_document", _fail_parse)

    stats = init_and_ingest(cfg=cfg)
    assert stats["skip_reasons"].get("empty_bytes", 0) == 1


def test_empty_parsed_document_is_skipped(tmp_path, monkeypatch):
    root = tmp_path
    f = root / "blank.txt"
    f.write_bytes(b"   \n  \t")

    cfg = default_project_config(root)
    cfg.storage.database_url = f"sqlite:///{(root / '.rag' / 'rag.db').resolve().as_posix()}"

    # Prevent heavy model loading
    monkeypatch.setattr("raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
                        lambda model_id: _dummy_embedder_factory())

    # Make parser return an empty-text document
    def _empty_parse(self, inp):
        doc = SimpleNamespace(doc_id=inp.doc_id, text="", source=inp.doc_id)
        return SimpleNamespace(document=doc)
    monkeypatch.setattr(ParserService, "parse_document", _empty_parse)

    # Ensure chunker is NOT called
    def _fail_chunk(self, conf):
        raise AssertionError(
            "chunker should not be called for empty parsed document")
    monkeypatch.setattr(Chunker, "chunk", _fail_chunk)

    stats = init_and_ingest(cfg=cfg)
    assert stats["skip_reasons"].get("empty_text_after_parse", 0) == 1


def test_whitespace_only_file_is_skipped(tmp_path, monkeypatch):
    root = tmp_path
    f = root / "ws.txt"
    f.write_bytes(b"   \n   \n")

    cfg = default_project_config(root)
    cfg.storage.database_url = f"sqlite:///{(root / '.rag' / 'rag.db').resolve().as_posix()}"

    monkeypatch.setattr("raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
                        lambda model_id: _dummy_embedder_factory())

    def _ws_parse(self, inp):
        doc = SimpleNamespace(
            doc_id=inp.doc_id, text="   \n  ", source=inp.doc_id)
        return SimpleNamespace(document=doc)
    monkeypatch.setattr(ParserService, "parse_document", _ws_parse)

    monkeypatch.setattr(Chunker, "chunk", lambda self, conf: (
        _ for _ in ()).throw(AssertionError("chunker should not be called")))

    stats = init_and_ingest(cfg=cfg)
    assert stats["skip_reasons"].get("empty_text_after_parse", 0) == 1
