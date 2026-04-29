"""
Tests for Step 3: EmbeddingConfigValidator + ModelCapabilityError.
"""
from __future__ import annotations

import pytest

from raggen.core.config.project import EmbeddingConfig
from raggen.core.embeddings.capabilities import ModelCapabilities
from raggen.core.embeddings.config_validator import (
    EmbeddingConfigValidator,
    ModelCapabilityError,
)

_CAPS_384 = ModelCapabilities(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    actual_dim=384,
    max_seq_length=256,
    max_batch_size=None,
)

_CAPS_768 = ModelCapabilities(
    model_id="sentence-transformers/all-mpnet-base-v2",
    actual_dim=768,
    max_seq_length=512,
    max_batch_size=None,
)


# ---------------------------------------------------------------------------
# No error cases
# ---------------------------------------------------------------------------


class TestNoError:
    def test_none_dim_is_accepted(self):
        """dim=None means auto-detect — validator must not raise."""
        cfg = EmbeddingConfig(dim=None)
        EmbeddingConfigValidator.validate(cfg, _CAPS_384)  # no exception

    def test_matching_dim_is_accepted(self):
        """dim pinned to the value the model actually returns — must not raise."""
        cfg = EmbeddingConfig(dim=384)
        EmbeddingConfigValidator.validate(cfg, _CAPS_384)  # no exception

    def test_matching_768_dim_is_accepted(self):
        cfg = EmbeddingConfig(dim=768)
        EmbeddingConfigValidator.validate(cfg, _CAPS_768)  # no exception


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestMismatchRaisesError:
    def test_raises_model_capability_error(self):
        """Mismatched dim must raise ModelCapabilityError."""
        cfg = EmbeddingConfig(dim=384)
        with pytest.raises(ModelCapabilityError):
            EmbeddingConfigValidator.validate(cfg, _CAPS_768)

    def test_is_runtime_error_subclass(self):
        """ModelCapabilityError must be a RuntimeError so CLI can catch it cleanly."""
        assert issubclass(ModelCapabilityError, RuntimeError)

    def test_error_message_contains_model_id(self):
        cfg = EmbeddingConfig(dim=384)
        with pytest.raises(ModelCapabilityError, match="all-mpnet-base-v2"):
            EmbeddingConfigValidator.validate(cfg, _CAPS_768)

    def test_error_message_contains_configured_dim(self):
        cfg = EmbeddingConfig(dim=384)
        with pytest.raises(ModelCapabilityError, match="384"):
            EmbeddingConfigValidator.validate(cfg, _CAPS_768)

    def test_error_message_contains_actual_dim(self):
        cfg = EmbeddingConfig(dim=384)
        with pytest.raises(ModelCapabilityError, match="768"):
            EmbeddingConfigValidator.validate(cfg, _CAPS_768)

    def test_error_message_contains_fix_hint(self):
        """The error must tell the user how to fix the config."""
        cfg = EmbeddingConfig(dim=384)
        with pytest.raises(ModelCapabilityError) as exc_info:
            EmbeddingConfigValidator.validate(cfg, _CAPS_768)
        msg = str(exc_info.value).lower()
        assert "config.toml" in msg or "fix" in msg

    def test_error_message_suggests_removing_line(self):
        """The error must mention removing the line as an alternative."""
        cfg = EmbeddingConfig(dim=384)
        with pytest.raises(ModelCapabilityError) as exc_info:
            EmbeddingConfigValidator.validate(cfg, _CAPS_768)
        msg = str(exc_info.value).lower()
        assert "remove" in msg or "automatic" in msg

    def test_reversed_mismatch_also_raises(self):
        """Config dim=768 but model returns 384 must also raise."""
        cfg = EmbeddingConfig(dim=768)
        with pytest.raises(ModelCapabilityError):
            EmbeddingConfigValidator.validate(cfg, _CAPS_384)


# ---------------------------------------------------------------------------
# Dim resolution contract (caller's responsibility, but tested for clarity)
# ---------------------------------------------------------------------------


class TestDimResolution:
    def test_none_dim_should_be_resolved_to_actual(self):
        """
        The validator accepts None; the caller resolves it via:
            cfg.embedding.dim = cfg.embedding.dim or caps.actual_dim
        Verify that pattern produces the expected integer.
        """
        cfg = EmbeddingConfig(dim=None)
        caps = _CAPS_384
        EmbeddingConfigValidator.validate(cfg, caps)
        resolved = cfg.dim or caps.actual_dim
        assert resolved == 384

    def test_pinned_dim_is_preserved_when_correct(self):
        """If dim is pinned and correct, resolution keeps it unchanged."""
        cfg = EmbeddingConfig(dim=384)
        caps = _CAPS_384
        EmbeddingConfigValidator.validate(cfg, caps)
        resolved = cfg.dim or caps.actual_dim
        assert resolved == 384
