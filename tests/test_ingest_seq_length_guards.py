"""
Tests for chunk_size vs model max_seq_length validation in do_ingest().

Option E behaviour:
- unit = "tokens": hard ConfigError at startup if chunk_size > max_seq_length - 2
- unit = "chars":  advisory warning at startup if chunk_size // 3 > max_seq_length - 2
"""
from __future__ import annotations

import numpy as np
import pytest

from raggen.core.config.project import ConfigError, default_project_config, GroupChunkingConfig
from raggen.core.embeddings.capabilities import ModelCapabilities
from raggen.core.embeddings.model_specs_cache import ModelSpecsCache
from raggen.core.ingest import do_ingest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MODEL_MAX = 64   # fake model limit — small enough to trigger easily in tests
_USABLE    = _MODEL_MAX - 2  # 62 content tokens


def _make_dummy_embedder(max_seq_length: int = _MODEL_MAX):
    """Return a factory that produces a minimal fake embedder with a known max_seq_length."""

    class _DummyEmbedder:
        model_id = "dummy"

        @property
        def max_seq_length(self):
            return max_seq_length

        @property
        def dim(self):
            return 4

        def embed_texts(self, texts, batch_size=32, normalize=True):
            return np.zeros((len(texts), 4), dtype=np.float32)

        def get_length_function(self):
            return len

    def _factory(*args, **kwargs):
        return _DummyEmbedder()

    return _factory


def _write_minimal_config(cfg_dir, cfg, dim: int = 4):
    """Write the minimum config.toml needed to bootstrap the ingest service.

    ``dim`` is written as 4 by default to match the dummy embedder's output,
    so the store's dimension check passes without a real model.
    """
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "config.toml").write_text(
        f"[storage]\n"
        f"backend_key = \"{cfg.storage.backend_key}\"\n"
        f"database_url = \"{cfg.storage.database_url}\"\n"
        f"vector_backend_import = \"{cfg.storage.vector_backend_import}\"\n"
        f"[embedding]\n"
        f"model_id = \"{cfg.embedding.model_id}\"\n"
        f"dim = {dim}\n"
        f"normalize = {str(cfg.embedding.normalize).lower()}\n"
    )


def _bootstrap_project(tmp_path, extra_toml: str = "") -> object:
    """Create a minimal project in tmp_path and return the bootstrapped config."""
    cfg = default_project_config(tmp_path)
    cfg.storage.database_url = (
        f"sqlite:///{(tmp_path / '.rag' / 'rag.db').resolve().as_posix()}"
    )
    cfg_dir = tmp_path / ".rag"
    _write_minimal_config(cfg_dir, cfg)

    if extra_toml:
        with open(cfg_dir / "config.toml", "a") as f:
            f.write("\n" + extra_toml)

    from raggen.core.bootstrap import bootstrap
    bootstrapped = bootstrap(cfg_dir / "config.toml")

    from raggen.core.store.initializer import init_database
    init_database(bootstrapped)

    # Seed the model specs cache so do_ingest() doesn't raise MissingModelSpecsError.
    # The dummy embedder uses dim=4 and the configured max_seq_length.
    specs_dir = tmp_path / ".rag" / "metadata" / "model_specs"
    ModelSpecsCache(specs_dir).put(
        ModelCapabilities(
            model_id=bootstrapped.embedding.model_id,
            actual_dim=4,
            max_seq_length=_MODEL_MAX,
            max_batch_size=None,
        )
    )

    return bootstrapped


# ---------------------------------------------------------------------------
# unit = "tokens" — hard ConfigError
# ---------------------------------------------------------------------------


class TestTokenUnitHardError:
    def test_chunk_size_exactly_at_limit_is_accepted(self, tmp_path, monkeypatch):
        """chunk_size == max_seq_length - 2 is the safe ceiling — must not raise."""
        _bootstrap_project(
            tmp_path,
            extra_toml=(
                "[chunking.fallback]\n"
                f"strategy = \"fixed\"\n"
                f"unit = \"tokens\"\n"
                f"chunk_size = {_USABLE}\n"
                f"overlap = 0\n"
            ),
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.create_embedder",
            _make_dummy_embedder(_MODEL_MAX),
        )

        # No files → ingest succeeds immediately without error
        result = do_ingest()
        assert result.success

    def test_chunk_size_one_over_limit_raises_config_error(self, tmp_path, monkeypatch):
        """chunk_size == max_seq_length - 1 (one over usable) must raise ConfigError."""
        _bootstrap_project(
            tmp_path,
            extra_toml=(
                "[chunking.fallback]\n"
                f"strategy = \"fixed\"\n"
                f"unit = \"tokens\"\n"
                f"chunk_size = {_USABLE + 1}\n"
                f"overlap = 0\n"
            ),
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.create_embedder",
            _make_dummy_embedder(_MODEL_MAX),
        )

        with pytest.raises(ConfigError, match="chunk_size"):
            do_ingest()

    def test_error_message_names_the_group(self, tmp_path, monkeypatch):
        """ConfigError message must mention the offending group name."""
        _bootstrap_project(
            tmp_path,
            extra_toml=(
                "[chunking.fallback]\n"
                f"strategy = \"fixed\"\n"
                f"unit = \"tokens\"\n"
                f"chunk_size = {_MODEL_MAX * 2}\n"
                f"overlap = 0\n"
            ),
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.create_embedder",
            _make_dummy_embedder(_MODEL_MAX),
        )

        with pytest.raises(ConfigError, match="'fallback'"):
            do_ingest()

    def test_error_raised_before_any_files_are_processed(self, tmp_path, monkeypatch):
        """ConfigError must fire at startup, not mid-ingest."""
        root = tmp_path
        (root / "doc.txt").write_text("some content here")

        _bootstrap_project(
            tmp_path,
            extra_toml=(
                "[chunking.fallback]\n"
                f"strategy = \"fixed\"\n"
                f"unit = \"tokens\"\n"
                f"chunk_size = {_MODEL_MAX * 10}\n"
                f"overlap = 0\n"
            ),
        )
        monkeypatch.chdir(root)
        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.create_embedder",
            _make_dummy_embedder(_MODEL_MAX),
        )

        # The error must propagate — ingest does not return a result envelope
        with pytest.raises(ConfigError):
            do_ingest()


# ---------------------------------------------------------------------------
# unit = "chars" — advisory estimation warning
# ---------------------------------------------------------------------------

# With _CHARS_PER_TOKEN_FLOOR = 3 and _USABLE = 62:
#   safe ceiling:   chunk_size // 3 must be < 62, so chunk_size <= 185
#   trigger point:  chunk_size // 3 must be > 62, so chunk_size >= 189
#
# Avoid the boundary (186–188) where integer division is ambiguous — use values
# clearly on each side so the tests don't depend on floor-division rounding.

_CHARS_SAFE    = (_USABLE - 1) * 3      # 183 → 183 // 3 = 61 < 62 — no warning
_CHARS_TRIGGER = (_USABLE + 1) * 3      # 189 → 189 // 3 = 63 > 62 — warning fires


class TestCharsUnitEstimationWarning:
    def test_safe_chunk_size_emits_no_warning(self, tmp_path, monkeypatch):
        """chunk_size at or below the estimated ceiling must not produce a warning."""
        _bootstrap_project(
            tmp_path,
            extra_toml=(
                "[chunking.fallback]\n"
                f"strategy = \"fixed\"\n"
                f"unit = \"chars\"\n"
                f"chunk_size = {_CHARS_SAFE}\n"
                f"overlap = 0\n"
            ),
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.create_embedder",
            _make_dummy_embedder(_MODEL_MAX),
        )

        result = do_ingest()
        estimate_warnings = [
            w for w in result.warnings if w.code == "chunk_size_estimate_warning"
        ]
        assert estimate_warnings == []

    def test_oversized_chunk_size_emits_warning(self, tmp_path, monkeypatch):
        """chunk_size above the estimated ceiling must emit an advisory warning."""
        _bootstrap_project(
            tmp_path,
            extra_toml=(
                "[chunking.fallback]\n"
                f"strategy = \"fixed\"\n"
                f"unit = \"chars\"\n"
                f"chunk_size = {_CHARS_TRIGGER}\n"
                f"overlap = 0\n"
            ),
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.create_embedder",
            _make_dummy_embedder(_MODEL_MAX),
        )

        result = do_ingest()
        estimate_warnings = [
            w for w in result.warnings if w.code == "chunk_size_estimate_warning"
        ]
        assert len(estimate_warnings) == 1

    def test_warning_message_mentions_group_and_estimation(self, tmp_path, monkeypatch):
        """Warning must name the group and make clear this is an estimate."""
        _bootstrap_project(
            tmp_path,
            extra_toml=(
                "[chunking.fallback]\n"
                f"strategy = \"fixed\"\n"
                f"unit = \"chars\"\n"
                f"chunk_size = {_CHARS_TRIGGER}\n"
                f"overlap = 0\n"
            ),
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.create_embedder",
            _make_dummy_embedder(_MODEL_MAX),
        )

        result = do_ingest()
        w = next(w for w in result.warnings if w.code == "chunk_size_estimate_warning")
        assert "fallback" in w.message
        assert "estimated" in w.message.lower() or "estimate" in w.message.lower()

    def test_warning_does_not_abort_ingest(self, tmp_path, monkeypatch):
        """An estimation warning must not prevent the ingest from completing."""
        root = tmp_path
        (root / "note.txt").write_text("hello world")

        _bootstrap_project(
            tmp_path,
            extra_toml=(
                "[chunking.fallback]\n"
                f"strategy = \"fixed\"\n"
                f"unit = \"chars\"\n"
                f"chunk_size = {_CHARS_TRIGGER}\n"
                f"overlap = 0\n"
            ),
        )
        monkeypatch.chdir(root)
        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.create_embedder",
            _make_dummy_embedder(_MODEL_MAX),
        )

        result = do_ingest()
        assert result.success
        assert result.data["summary"]["docs_parsed"] == 1
