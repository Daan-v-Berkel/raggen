"""
Integration tests for real embedding backend implementations.

These tests load actual models and produce real embeddings.
They are marked `integration` because they require a network connection
on the first run to download the model (~24 MB).

Run with:
    pytest -m integration tests/test_embedder_backends.py

ONNX tests always run when fastembed is installed (core dependency).
Torch tests are skipped unless sentence-transformers is installed:
    pip install 'raggen[torch]'
"""
from __future__ import annotations

import numpy as np
import pytest

# Small model used by both backends to allow sharing the CI model cache.
_MODEL_ID = "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# ONNX backend (always available — fastembed is a core dep)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def onnx_embedder():
    pytest.importorskip("fastembed")
    from raggen.core.embeddings.backends.onnx import OnnxEmbedder
    return OnnxEmbedder(model_id=_MODEL_ID, batch_size=4, normalize=True)


@pytest.mark.integration
def test_onnx_embed_single_text_shape(onnx_embedder):
    result = onnx_embedder.embed_texts(["hello world"])
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, onnx_embedder.dim)


@pytest.mark.integration
def test_onnx_embed_multiple_texts_row_count(onnx_embedder):
    texts = ["foo", "bar", "baz"]
    result = onnx_embedder.embed_texts(texts)
    assert result.shape[0] == len(texts)
    assert result.shape[1] == onnx_embedder.dim


@pytest.mark.integration
def test_onnx_dim_matches_output(onnx_embedder):
    result = onnx_embedder.embed_texts(["test"])
    assert result.shape[1] == onnx_embedder.dim


@pytest.mark.integration
def test_onnx_max_seq_length_is_positive(onnx_embedder):
    assert isinstance(onnx_embedder.max_seq_length, int)
    assert onnx_embedder.max_seq_length > 0


@pytest.mark.integration
def test_onnx_length_function_returns_int(onnx_embedder):
    length_fn = onnx_embedder.get_length_function()
    assert callable(length_fn)
    result = length_fn("hello world")
    assert isinstance(result, int)
    assert result > 0


# ---------------------------------------------------------------------------
# Torch backend (optional — needs sentence-transformers)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def torch_embedder():
    pytest.importorskip("sentence_transformers")
    from raggen.core.embeddings.backends.torch import TorchEmbedder
    return TorchEmbedder(model_id=_MODEL_ID, batch_size=4, normalize=True)


@pytest.mark.integration
@pytest.mark.requires_torch
def test_torch_embed_single_text_shape(torch_embedder):
    result = torch_embedder.embed_texts(["hello world"])
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, torch_embedder.dim)


@pytest.mark.integration
@pytest.mark.requires_torch
def test_torch_embed_multiple_texts_row_count(torch_embedder):
    texts = ["foo", "bar", "baz"]
    result = torch_embedder.embed_texts(texts)
    assert result.shape[0] == len(texts)
    assert result.shape[1] == torch_embedder.dim


@pytest.mark.integration
@pytest.mark.requires_torch
def test_torch_dim_matches_output(torch_embedder):
    result = torch_embedder.embed_texts(["test"])
    assert result.shape[1] == torch_embedder.dim


@pytest.mark.integration
@pytest.mark.requires_torch
def test_torch_max_seq_length_is_positive(torch_embedder):
    assert isinstance(torch_embedder.max_seq_length, int)
    assert torch_embedder.max_seq_length > 0


@pytest.mark.integration
@pytest.mark.requires_torch
def test_torch_length_function_returns_int(torch_embedder):
    length_fn = torch_embedder.get_length_function()
    assert callable(length_fn)
    result = length_fn("hello world")
    assert isinstance(result, int)
    assert result > 0


# ---------------------------------------------------------------------------
# Cross-backend: same model, same input → same output dimension
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_torch
def test_onnx_and_torch_produce_same_dim(onnx_embedder, torch_embedder):
    assert onnx_embedder.dim == torch_embedder.dim
