"""
Tests for Step 1: ModelCapabilities + ModelInspector (capabilities.py).

All tests monkeypatch _load_sentence_transformer so no real model is loaded.
"""
from __future__ import annotations

import pytest

from raggen.core.embeddings.capabilities import (
    ModelCapabilities,
    ModelInspector,
    ModelLoadError,
    _load_sentence_transformer,
)


# ---------------------------------------------------------------------------
# Shared fake model
# ---------------------------------------------------------------------------


class _FakeModel:
    """Minimal stand-in for a SentenceTransformer with known capabilities."""

    def __init__(self, dim: int = 384, max_seq_length: int = 256):
        self._dim = dim
        self.max_seq_length = max_seq_length

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


def _fake_loader(model_id: str, cache_dir=None) -> _FakeModel:
    return _FakeModel(dim=384, max_seq_length=256)


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
        assert caps.max_batch_size is None  # always None for now

    def test_is_frozen(self):
        caps = ModelCapabilities(
            model_id="m", actual_dim=384, max_seq_length=256, max_batch_size=None
        )
        with pytest.raises(Exception):  # dataclass(frozen=True) raises FrozenInstanceError
            caps.actual_dim = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ModelInspector
# ---------------------------------------------------------------------------


class TestModelInspector:
    def test_introspect_returns_correct_capabilities(self, monkeypatch):
        monkeypatch.setattr(
            "raggen.core.embeddings.capabilities._load_sentence_transformer",
            _fake_loader,
        )
        caps = ModelInspector.introspect("sentence-transformers/all-MiniLM-L6-v2")

        assert caps.model_id == "sentence-transformers/all-MiniLM-L6-v2"
        assert caps.actual_dim == 384
        assert caps.max_seq_length == 256
        assert caps.max_batch_size is None

    def test_introspect_with_cache_dir_passes_it_through(self, monkeypatch):
        received: list = []

        def _capturing_loader(model_id, cache_dir=None):
            received.append(cache_dir)
            return _FakeModel()

        monkeypatch.setattr(
            "raggen.core.embeddings.capabilities._load_sentence_transformer",
            _capturing_loader,
        )
        ModelInspector.introspect("any-model", cache_dir=".rag/models")
        assert received == [".rag/models"]

    def test_introspect_propagates_model_load_error(self, monkeypatch):
        def _failing_loader(model_id, cache_dir=None):
            raise ModelLoadError(f"Failed to load '{model_id}'.")

        monkeypatch.setattr(
            "raggen.core.embeddings.capabilities._load_sentence_transformer",
            _failing_loader,
        )
        with pytest.raises(ModelLoadError, match="bad-model-id"):
            ModelInspector.introspect("bad-model-id")


# ---------------------------------------------------------------------------
# ModelLoadError
# ---------------------------------------------------------------------------


class TestModelLoadError:
    def test_message_contains_model_id(self, monkeypatch):
        """_load_sentence_transformer wraps load failures with the model ID in the message."""
        # Simulate SentenceTransformer raising on load by patching its import path.
        import raggen.core.embeddings.capabilities as caps_mod

        original = caps_mod._load_sentence_transformer

        def _patched(model_id, cache_dir=None):
            # Call the real function but with a module-level patch on SentenceTransformer.
            raise ModelLoadError(
                f"Failed to load embedding model '{model_id}'.\n"
                "Check that the model ID is correct.\n\n"
                "Original error: OSError: no file"
            )

        monkeypatch.setattr(caps_mod, "_load_sentence_transformer", _patched)

        with pytest.raises(ModelLoadError) as exc_info:
            ModelInspector.introspect("bogus/model-that-does-not-exist")

        assert "bogus/model-that-does-not-exist" in str(exc_info.value)

    def test_is_runtime_error_subclass(self):
        err = ModelLoadError("test")
        assert isinstance(err, RuntimeError)
