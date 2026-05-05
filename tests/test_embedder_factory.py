"""
Unit tests for the embedder factory and backend detection.

These tests do NOT load real models — they verify routing logic only.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


def test_create_embedder_unknown_backend_raises():
    from raggen.core.config.project import ConfigError
    from raggen.core.embeddings.embedder import create_embedder

    with pytest.raises(ConfigError, match="Unknown embedding backend"):
        create_embedder("some-model", backend="bad_value")


def test_create_embedder_onnx_returns_onnx_embedder():
    pytest.importorskip("fastembed")
    from raggen.core.embeddings.backends.onnx import OnnxEmbedder
    from raggen.core.embeddings.embedder import create_embedder

    # Patch model init so we don't download anything
    with patch.object(OnnxEmbedder, "__init__", return_value=None):
        embedder = create_embedder("test-model", backend="onnx")

    assert isinstance(embedder, OnnxEmbedder)


def test_create_embedder_torch_returns_torch_embedder():
    pytest.importorskip("sentence_transformers")
    from raggen.core.embeddings.backends.torch import TorchEmbedder
    from raggen.core.embeddings.embedder import create_embedder

    with patch.object(TorchEmbedder, "__init__", return_value=None):
        embedder = create_embedder("test-model", backend="torch")

    assert isinstance(embedder, TorchEmbedder)


def test_detect_backend_returns_onnx_when_fastembed_installed():
    import importlib.util

    fastembed_spec = importlib.util.find_spec("fastembed")
    if fastembed_spec is None:
        pytest.skip("fastembed not installed")

    from raggen.core.embeddings.embedder import _detect_backend

    assert _detect_backend() == "onnx"


def test_detect_backend_returns_torch_when_only_torch_installed():
    import importlib.util as _importlib_util
    from raggen.core.embeddings.embedder import _detect_backend

    _real_find_spec = _importlib_util.find_spec

    def fake_find_spec(name):
        if name == "fastembed":
            return None
        # Delegate to the real find_spec for everything else to avoid recursion.
        return _real_find_spec(name)

    with patch("raggen.core.embeddings.embedder.importlib.util.find_spec", side_effect=fake_find_spec):
        if _real_find_spec("sentence_transformers") is None:
            with pytest.raises(RuntimeError, match="No embedding backend"):
                _detect_backend()
        else:
            assert _detect_backend() == "torch"


def test_detect_backend_raises_when_nothing_installed():
    from raggen.core.embeddings.embedder import _detect_backend

    with patch("raggen.core.embeddings.embedder.importlib.util.find_spec", return_value=None):
        with pytest.raises(RuntimeError, match="No embedding backend is installed"):
            _detect_backend()


def test_create_embedder_auto_resolves_to_onnx_when_fastembed_present():
    pytest.importorskip("fastembed")
    from raggen.core.embeddings.backends.onnx import OnnxEmbedder
    from raggen.core.embeddings.embedder import create_embedder

    with patch.object(OnnxEmbedder, "__init__", return_value=None):
        embedder = create_embedder("test-model", backend="auto")

    assert isinstance(embedder, OnnxEmbedder)
