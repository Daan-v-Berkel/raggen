"""
Tests for ModelCapabilities + ModelInspector (capabilities.py).

ModelInspector.introspect() delegates to create_embedder(), so tests patch
that factory rather than any backend-specific loader.
"""
from __future__ import annotations

import pytest

from raggen.core.embeddings.capabilities import (
    ModelCapabilities,
    ModelInspector,
    ModelLoadError,
)


# ---------------------------------------------------------------------------
# Shared fake embedder (satisfies EmbedderBackend protocol)
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Minimal stand-in for any EmbedderBackend with known capabilities."""

    def __init__(self, dim: int = 384, max_seq_length: int = 256):
        self.model_id = "test-model"
        self.batch_size = 32
        self.normalize = True
        self._dim = dim
        self._max_seq_length = max_seq_length

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    def get_length_function(self):
        return len

    def embed_texts(self, texts, batch_size=None, normalize=None):
        import numpy as np
        return np.zeros((len(texts), self._dim), dtype="float32")


def _fake_create_embedder(model_id, backend="auto", cache_dir=None, **kwargs):
    return _FakeEmbedder(dim=384, max_seq_length=256)


# ---------------------------------------------------------------------------
# ModelCapabilities
# ---------------------------------------------------------------------------


class TestModelCapabilities:
    def test_all_fields_populated(self):
        caps = ModelCapabilities(
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            actual_dim=384,
            max_seq_length=256,
            max_batch_size=None,
        )
        assert caps.model_id == "sentence-transformers/all-MiniLM-L6-v2"
        assert caps.actual_dim == 384
        assert caps.max_seq_length == 256
        assert caps.max_batch_size is None

    def test_is_frozen(self):
        caps = ModelCapabilities(
            model_id="m", actual_dim=384, max_seq_length=256, max_batch_size=None
        )
        with pytest.raises(Exception):
            caps.actual_dim = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ModelInspector
# ---------------------------------------------------------------------------


class TestModelInspector:
    def test_introspect_returns_correct_capabilities(self, monkeypatch):
        monkeypatch.setattr(
            "raggen.core.embeddings.capabilities.create_embedder",
            _fake_create_embedder,
        )
        caps = ModelInspector.introspect("sentence-transformers/all-MiniLM-L6-v2")

        assert caps.model_id == "sentence-transformers/all-MiniLM-L6-v2"
        assert caps.actual_dim == 384
        assert caps.max_seq_length == 256
        assert caps.max_batch_size is None

    def test_introspect_with_cache_dir_passes_it_through(self, monkeypatch):
        received: list = []

        def _capturing_factory(model_id, backend="auto", cache_dir=None, **kwargs):
            received.append(cache_dir)
            return _FakeEmbedder()

        monkeypatch.setattr(
            "raggen.core.embeddings.capabilities.create_embedder",
            _capturing_factory,
        )
        ModelInspector.introspect("any-model", cache_dir=".rag/models")
        assert received == [".rag/models"]

    def test_introspect_propagates_model_load_error(self, monkeypatch):
        def _failing_factory(model_id, **kwargs):
            raise ModelLoadError(f"Failed to load '{model_id}'.")

        monkeypatch.setattr(
            "raggen.core.embeddings.capabilities.create_embedder",
            _failing_factory,
        )
        with pytest.raises(ModelLoadError, match="bad-model-id"):
            ModelInspector.introspect("bad-model-id")


# ---------------------------------------------------------------------------
# ModelLoadError
# ---------------------------------------------------------------------------


class TestModelLoadError:
    def test_message_contains_model_id(self, monkeypatch):
        def _patched(model_id, **kwargs):
            raise ModelLoadError(
                f"Failed to load embedding model '{model_id}'.\n"
                "Check that the model ID is correct.\n\n"
                "Original error: OSError: no file"
            )

        monkeypatch.setattr(
            "raggen.core.embeddings.capabilities.create_embedder",
            _patched,
        )

        with pytest.raises(ModelLoadError) as exc_info:
            ModelInspector.introspect("bogus/model-that-does-not-exist")

        assert "bogus/model-that-does-not-exist" in str(exc_info.value)

    def test_is_runtime_error_subclass(self):
        err = ModelLoadError("test")
        assert isinstance(err, RuntimeError)
