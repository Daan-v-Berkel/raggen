"""
Tests for Step 7: ProjectValidator — unified validation entry point.

Covers:
  - validate_for_build: dim mismatch, correct config, destructive skips schema check
  - validate_for_ingest: schema mismatch on BREAKING change, chunking drift,
    chunk-size limit, fresh project, clean run
  - End-to-end: do_ingest raises SchemaMismatchError after config drift
"""
from __future__ import annotations

import numpy as np
import pytest

from raggen.core.config.project import GroupChunkingConfig, default_project_config
from raggen.core.embeddings.capabilities import ModelCapabilities
from raggen.core.embeddings.config_validator import ModelCapabilityError
from raggen.core.embeddings.model_specs_cache import ModelSpecsCache
from raggen.core.ingest import do_ingest
from raggen.core.metadata.models import ProjectLifecycleState
from raggen.core.metadata.store import (
    create_project_state,
    save_project_state,
)
from raggen.core.store.exceptions import SchemaMismatchError
from raggen.core.validation.project_validator import ProjectValidator

_DUMMY_DIM = 4
_DUMMY_MAX_SEQ = 512  # large enough not to warn on default chunk_size=1000 chars


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _caps(dim: int = _DUMMY_DIM, max_seq: int = _DUMMY_MAX_SEQ) -> ModelCapabilities:
    return ModelCapabilities(
        model_id="test/model",
        actual_dim=dim,
        max_seq_length=max_seq,
        max_batch_size=None,
    )


def _seed_model_specs_cache(root, model_id, dim=_DUMMY_DIM, max_seq=_DUMMY_MAX_SEQ):
    ModelSpecsCache(root / ".rag" / "metadata" / "model_specs").put(
        ModelCapabilities(
            model_id=model_id,
            actual_dim=dim,
            max_seq_length=max_seq,
            max_batch_size=None,
        )
    )


def _dummy_embedder_factory(*a, **kw):
    class _D:
        model_id = "dummy"

        @property
        def dim(self):
            return _DUMMY_DIM

        @property
        def max_seq_length(self):
            return _DUMMY_MAX_SEQ

        def embed_texts(self, texts, batch_size=32, normalize=True):
            return np.zeros((len(texts), _DUMMY_DIM), dtype=np.float32)

        def get_length_function(self):
            return len

    return _D()


def _bootstrap_project(tmp_path, monkeypatch):
    """Bootstrap a minimal project, seed the specs cache, and save SET_UP state."""
    root = tmp_path
    monkeypatch.chdir(root)

    cfg = default_project_config(root)
    cfg.storage.database_url = (
        f"sqlite:///{(root / '.rag' / 'rag.db').resolve().as_posix()}"
    )

    cfg_dir = root / ".rag"
    cfg_dir.mkdir(exist_ok=True)
    cfg_file = cfg_dir / "config.toml"
    cfg_file.write_text(
        f"[storage]\n"
        f"backend_key = \"{cfg.storage.backend_key}\"\n"
        f"database_url = \"{cfg.storage.database_url}\"\n"
        f"vector_backend_import = \"{cfg.storage.vector_backend_import}\"\n"
        f"[embedding]\n"
        f"model_id = \"{cfg.embedding.model_id}\"\n"
        f"dim = {_DUMMY_DIM}\n"
        f"normalize = {str(cfg.embedding.normalize).lower()}\n"
    )

    from raggen.core.bootstrap import bootstrap
    cfg = bootstrap(cfg_file)

    from raggen.core.store.initializer import init_database
    init_database(cfg)

    _seed_model_specs_cache(root, cfg.embedding.model_id)

    state = create_project_state(cfg=cfg, state=ProjectLifecycleState.SET_UP)
    save_project_state(state)

    return cfg


# ---------------------------------------------------------------------------
# validate_for_build
# ---------------------------------------------------------------------------


class TestValidateForBuild:
    def test_correct_dim_passes(self, tmp_path, cfg_factory, write_cfg):
        """Correct pinned dim must not raise."""
        cfg = cfg_factory(tmp_path, embedding_dim=_DUMMY_DIM)
        cfg_path = write_cfg(cfg, tmp_path)
        from raggen.core.bootstrap import bootstrap
        from raggen.core.store.initializer import init_database
        from raggen.core.runtime import get_engine
        cfg = bootstrap(cfg_path)
        init_database(cfg)
        engine = get_engine()
        caps = _caps(dim=_DUMMY_DIM)
        ProjectValidator.validate_for_build(cfg, caps, engine)  # must not raise

    def test_none_dim_resolves_to_caps_actual_dim(self, tmp_path, cfg_factory, write_cfg):
        """dim=None must be resolved to caps.actual_dim as a side effect."""
        # Init DB with dim=768 so the schema check after resolution sees no mismatch.
        cfg = cfg_factory(tmp_path, embedding_dim=768)
        cfg_path = write_cfg(cfg, tmp_path)
        from raggen.core.bootstrap import bootstrap
        from raggen.core.store.initializer import init_database
        from raggen.core.runtime import get_engine
        cfg = bootstrap(cfg_path)
        init_database(cfg)
        cfg.embedding.dim = None  # simulate unset dim
        engine = get_engine()
        caps = _caps(dim=768)
        ProjectValidator.validate_for_build(cfg, caps, engine)
        assert cfg.embedding.dim == 768  # resolved by validator

    def test_wrong_dim_raises_model_capability_error(
        self, tmp_path, cfg_factory, write_cfg
    ):
        """Pinned dim contradicting model capabilities must raise ModelCapabilityError."""
        cfg = cfg_factory(tmp_path, embedding_dim=384)
        cfg_path = write_cfg(cfg, tmp_path)
        from raggen.core.bootstrap import bootstrap
        from raggen.core.store.initializer import init_database
        from raggen.core.runtime import get_engine
        cfg = bootstrap(cfg_path)
        init_database(cfg)
        engine = get_engine()
        caps = _caps(dim=768)
        with pytest.raises(ModelCapabilityError):
            ProjectValidator.validate_for_build(cfg, caps, engine)

    def test_breaking_schema_change_raises_schema_mismatch(
        self, tmp_path, cfg_factory, write_cfg
    ):
        """A BREAKING config change must raise SchemaMismatchError (non-destructive)."""
        cfg1 = cfg_factory(tmp_path, embedding_dim=1234)
        cfg_path = write_cfg(cfg1, tmp_path)
        from raggen.core.bootstrap import bootstrap
        from raggen.core.store.initializer import init_database
        from raggen.core.runtime import get_engine
        cfg1 = bootstrap(cfg_path)
        init_database(cfg1)

        cfg2 = cfg_factory(tmp_path, embedding_dim=9999)
        engine = get_engine()
        caps = _caps(dim=9999)
        with pytest.raises(SchemaMismatchError):
            ProjectValidator.validate_for_build(cfg2, caps, engine, destructive=False)

    def test_destructive_skips_schema_mismatch_check(
        self, tmp_path, cfg_factory, write_cfg
    ):
        """destructive=True must skip validate_existing_project so a rebuild can proceed."""
        cfg1 = cfg_factory(tmp_path, embedding_dim=1234)
        cfg_path = write_cfg(cfg1, tmp_path)
        from raggen.core.bootstrap import bootstrap
        from raggen.core.store.initializer import init_database
        from raggen.core.runtime import get_engine
        cfg1 = bootstrap(cfg_path)
        init_database(cfg1)

        cfg2 = cfg_factory(tmp_path, embedding_dim=9999)
        engine = get_engine()
        caps = _caps(dim=9999)
        # must NOT raise even though config differs from stored schema
        ProjectValidator.validate_for_build(cfg2, caps, engine, destructive=True)


# ---------------------------------------------------------------------------
# validate_for_ingest
# ---------------------------------------------------------------------------


class TestValidateForIngest:
    def _engine_and_state(self, tmp_path, monkeypatch):
        """Bootstrap a project and return (cfg, engine, state, caps)."""
        cfg = _bootstrap_project(tmp_path, monkeypatch)
        from raggen.core.runtime import get_engine
        from raggen.core.metadata.store import load_project_state
        engine = get_engine()
        state = load_project_state(cfg.project_root)
        caps = _caps()
        return cfg, engine, state, caps

    def test_clean_project_returns_no_warnings(self, tmp_path, monkeypatch):
        """Fresh project with no drift must return an empty warning list."""
        cfg, engine, state, caps = self._engine_and_state(tmp_path, monkeypatch)
        warnings = ProjectValidator.validate_for_ingest(cfg, engine, state, caps)
        assert warnings == []

    def test_breaking_schema_change_raises(self, tmp_path, monkeypatch):
        """SchemaMismatchError must be raised when a BREAKING field changed."""
        cfg, engine, state, caps = self._engine_and_state(tmp_path, monkeypatch)
        # Flip normalize — stored as True in DB, now False → BREAKING
        cfg.embedding.normalize = not cfg.embedding.normalize
        with pytest.raises(SchemaMismatchError):
            ProjectValidator.validate_for_ingest(cfg, engine, state, caps)

    def test_breaking_schema_change_message_contains_command(
        self, tmp_path, monkeypatch
    ):
        """SchemaMismatchError message must include 'rag build --destructive'."""
        cfg, engine, state, caps = self._engine_and_state(tmp_path, monkeypatch)
        cfg.embedding.normalize = not cfg.embedding.normalize
        with pytest.raises(SchemaMismatchError) as exc_info:
            ProjectValidator.validate_for_ingest(cfg, engine, state, caps)
        assert "rag build --destructive" in str(exc_info.value)

    def test_chunking_drift_returns_warning(self, tmp_path, monkeypatch):
        """Chunking drift must appear in the returned warnings list."""
        cfg, engine, state, caps = self._engine_and_state(tmp_path, monkeypatch)
        cfg.chunking["fallback"].chunk_size += 500
        warnings = ProjectValidator.validate_for_ingest(cfg, engine, state, caps)
        codes = [w.code for w in warnings]
        assert "chunking_drift" in codes

    def test_no_project_state_skips_drift_check(self, tmp_path, monkeypatch):
        """Passing state=None must skip drift detection (no warnings returned)."""
        cfg, engine, _, caps = self._engine_and_state(tmp_path, monkeypatch)
        cfg.chunking["fallback"].chunk_size += 500
        warnings = ProjectValidator.validate_for_ingest(cfg, engine, None, caps)
        drift = [w for w in warnings if w.code == "chunking_drift"]
        assert drift == []

    def test_token_chunk_size_over_limit_raises_config_error(
        self, tmp_path, monkeypatch
    ):
        """chunk_size > model capacity (tokens unit) must raise ConfigError."""
        from raggen.core.config.project import ConfigError
        cfg, engine, state, caps = self._engine_and_state(tmp_path, monkeypatch)
        cfg.chunking["fallback"].unit = "tokens"
        cfg.chunking["fallback"].chunk_size = caps.max_seq_length * 10
        with pytest.raises(ConfigError, match="chunk_size"):
            ProjectValidator.validate_for_ingest(cfg, engine, state, caps)

    def test_chars_chunk_size_estimate_warning(self, tmp_path, monkeypatch):
        """Oversized chars chunk_size must add a chunk_size_estimate_warning."""
        cfg, engine, state, caps = self._engine_and_state(tmp_path, monkeypatch)
        # 3 * (max_seq - 2 + 1) chars → estimated tokens exceed usable limit
        cfg.chunking["fallback"].unit = "chars"
        cfg.chunking["fallback"].chunk_size = (caps.max_seq_length + 1) * 3
        warnings = ProjectValidator.validate_for_ingest(cfg, engine, state, caps)
        codes = [w.code for w in warnings]
        assert "chunk_size_estimate_warning" in codes


# ---------------------------------------------------------------------------
# End-to-end: do_ingest raises after BREAKING config drift
# ---------------------------------------------------------------------------


class TestDoIngestBreakingDriftDetection:
    def test_breaking_config_change_raises_on_ingest(self, tmp_path, monkeypatch):
        """
        After a build, changing a BREAKING field (normalize) and running
        do_ingest() must raise SchemaMismatchError before any files are
        processed.
        """
        cfg = _bootstrap_project(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
            lambda *a, **kw: _dummy_embedder_factory(),
        )

        # Flip normalize — BREAKING change vs stored schema
        cfg.embedding.normalize = not cfg.embedding.normalize

        with pytest.raises(SchemaMismatchError):
            do_ingest()

    def test_runtime_config_change_does_not_raise_on_ingest(
        self, tmp_path, monkeypatch
    ):
        """
        Changing a RUNTIME field (query.model_id) must not raise — ingest
        should succeed without any schema-mismatch error.
        """
        cfg = _bootstrap_project(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
            lambda *a, **kw: _dummy_embedder_factory(),
        )

        cfg.query.model_id = "some-different-query-model"  # RUNTIME tier

        result = do_ingest()
        assert result.success

    def test_fresh_build_then_ingest_no_errors_no_warnings(
        self, tmp_path, monkeypatch
    ):
        """
        A fresh build followed immediately by ingest must produce no errors
        and no validation-related warnings.
        """
        cfg = _bootstrap_project(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
            lambda *a, **kw: _dummy_embedder_factory(),
        )

        result = do_ingest()
        assert result.success
        validation_codes = {"chunking_drift", "chunk_size_estimate_warning",
                            "schema_mismatch"}
        warning_codes = {w.code for w in result.warnings}
        assert warning_codes.isdisjoint(validation_codes)
