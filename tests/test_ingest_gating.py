from types import SimpleNamespace
import numpy as np

from raggen.core.config.project import default_project_config
from raggen.core.parsing.parser import ParserService
from raggen.core.chunking.chunker import ChunkerRegistry
from raggen.core.ingest import do_ingest


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

    # write complete config and bootstrap so runtime uses this config
    cfg_dir = root / '.rag'
    cfg_dir.mkdir(exist_ok=True)
    cfg_file = cfg_dir / 'config.toml'
    cfg_file.write_text(
        f"[storage]\n"
        f"backend_key = \"{cfg.storage.backend_key}\"\n"
        f"database_url = \"{cfg.storage.database_url}\"\n"
        f"vector_backend_import = \"{cfg.storage.vector_backend_import}\"\n"
        f"[embedding]\n"
        f"model_id = \"{cfg.embedding.model_id}\"\n"
        f"dim = {cfg.embedding.dim}\n"
        f"normalize = {str(cfg.embedding.normalize).lower()}\n"
    )
    from raggen.core.bootstrap import bootstrap
    cfg = bootstrap(cfg_file)
    from raggen.core.store.initializer import init_database
    init_database(cfg)

    # Prevent heavy model loading
    monkeypatch.chdir(root)
    monkeypatch.setattr("raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
                        lambda model_id: _dummy_embedder_factory())

    # Ensure parser is NOT called
    def _fail_parse(self, inp):
        raise AssertionError("parser should not be called for empty raw files")
    monkeypatch.setattr(ParserService, "parse_document", _fail_parse)

    result = do_ingest()
    empty_after_parse = [x for x in result.warnings if x.code == 'zero_bytes']
    assert len(empty_after_parse) == 1


def test_empty_parsed_document_is_skipped(tmp_path, monkeypatch):
    root = tmp_path
    f = root / "blank.txt"
    f.write_bytes(b"   \n  \t")

    cfg = default_project_config(root)
    cfg.storage.database_url = f"sqlite:///{(root / '.rag' / 'rag.db').resolve().as_posix()}"

    # write complete config and bootstrap so runtime uses this config
    cfg_dir = root / '.rag'
    cfg_dir.mkdir(exist_ok=True)
    cfg_file = cfg_dir / 'config.toml'
    cfg_file.write_text(
        f"[storage]\n"
        f"backend_key = \"{cfg.storage.backend_key}\"\n"
        f"database_url = \"{cfg.storage.database_url}\"\n"
        f"vector_backend_import = \"{cfg.storage.vector_backend_import}\"\n"
        f"[embedding]\n"
        f"model_id = \"{cfg.embedding.model_id}\"\n"
        f"dim = {cfg.embedding.dim}\n"
        f"normalize = {str(cfg.embedding.normalize).lower()}\n"
    )
    from raggen.core.bootstrap import bootstrap
    cfg = bootstrap(cfg_file)
    from raggen.core.store.initializer import init_database
    init_database(cfg)

    # Prevent heavy model loading
    monkeypatch.chdir(root)
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
    chunker = ChunkerRegistry().get("fallback")
    monkeypatch.setattr(chunker, "chunk", _fail_chunk)

    result = do_ingest()
    empty_after_parse = [x for x in result.warnings if x.code == 'empty_file']
    assert len(empty_after_parse) == 1


def test_whitespace_only_file_is_skipped(tmp_path, monkeypatch):
    root = tmp_path
    f = root / "ws.txt"
    f.write_bytes(b"   \n   \n")

    cfg = default_project_config(root)
    cfg.storage.database_url = f"sqlite:///{(root / '.rag' / 'rag.db').resolve().as_posix()}"

    # write complete config and bootstrap so runtime uses this config
    cfg_dir = root / '.rag'
    cfg_dir.mkdir(exist_ok=True)
    cfg_file = cfg_dir / 'config.toml'
    cfg_file.write_text(
        f"[storage]\n"
        f"backend_key = \"{cfg.storage.backend_key}\"\n"
        f"database_url = \"{cfg.storage.database_url}\"\n"
        f"vector_backend_import = \"{cfg.storage.vector_backend_import}\"\n"
        f"[embedding]\n"
        f"model_id = \"{cfg.embedding.model_id}\"\n"
        f"dim = {cfg.embedding.dim}\n"
        f"normalize = {str(cfg.embedding.normalize).lower()}\n"
    )
    from raggen.core.bootstrap import bootstrap
    cfg = bootstrap(cfg_file)
    from raggen.core.store.initializer import init_database
    init_database(cfg)
    # ensure project root lookup uses tmp_path
    monkeypatch.chdir(root)

    monkeypatch.setattr("raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
                        lambda model_id: _dummy_embedder_factory())

    def _ws_parse(self, inp):
        doc = SimpleNamespace(
            doc_id=inp.doc_id, text="   \n  ", source=inp.doc_id)
        return SimpleNamespace(document=doc)
    monkeypatch.setattr(ParserService, "parse_document", _ws_parse)

    chunker = ChunkerRegistry().get("fallback")
    monkeypatch.setattr(chunker, "chunk", lambda self, conf: (
        _ for _ in ()).throw(AssertionError("chunker should not be called")))

    result = do_ingest()
    empty_after_parse = [x for x in result.warnings if x.code == 'empty_file']
    assert len(empty_after_parse) == 1
