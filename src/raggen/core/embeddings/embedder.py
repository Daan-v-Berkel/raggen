from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Sequence

import numpy as np

if TYPE_CHECKING:
    from raggen.core.embeddings.backends.base import EmbedderBackend


@dataclass(frozen=True)
class EmbeddingResult:
    chunk_id: str
    vector: np.ndarray  # shape: (dim,)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def _detect_backend() -> str:
    """Return 'onnx' or 'torch' based on what is installed, preferring onnx."""
    if importlib.util.find_spec("fastembed") is not None:
        return "onnx"
    if importlib.util.find_spec("sentence_transformers") is not None:
        return "torch"
    raise RuntimeError(
        "No embedding backend is installed.\n"
        "  Default (ONNX):  pip install raggen\n"
        "  Torch backend:   pip install 'raggen[torch]'"
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_embedder(
    model_id: str,
    backend: str = "auto",
    cache_dir: str | None = None,
    batch_size: int = 32,
    normalize: bool = True,
) -> "EmbedderBackend":
    """Create and return an embedding backend instance.

    backend:
        "auto"  — use ONNX if fastembed is installed, fall back to torch.
        "onnx"  — always use fastembed (raises if not installed).
        "torch" — always use sentence-transformers (raises if not installed).
    """
    resolved = backend if backend != "auto" else _detect_backend()

    if resolved == "onnx":
        from raggen.core.embeddings.backends.onnx import OnnxEmbedder  # noqa: PLC0415
        return OnnxEmbedder(
            model_id=model_id,
            cache_dir=cache_dir,
            batch_size=batch_size,
            normalize=normalize,
        )

    if resolved == "torch":
        from raggen.core.embeddings.backends.torch import TorchEmbedder  # noqa: PLC0415
        return TorchEmbedder(
            model_id=model_id,
            cache_dir=cache_dir,
            batch_size=batch_size,
            normalize=normalize,
        )

    from raggen.core.config.project import ConfigError
    raise ConfigError(
        f"Unknown embedding backend: {resolved!r}. "
        "Valid values are 'auto', 'onnx', and 'torch'."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def embed_chunks(
    embedder: "EmbedderBackend",
    chunks: Sequence,
) -> List[EmbeddingResult]:
    """Embed a sequence of chunks, returning one EmbeddingResult per chunk.

    chunks must have:
      - chunk.chunk_id (str)
      - chunk.text (str)
    """
    texts = [getattr(ch, "text") for ch in chunks]
    ids = [getattr(ch, "chunk_id") for ch in chunks]

    matrix = embedder.embed_texts(texts)

    if matrix.shape[0] != len(ids):
        raise RuntimeError("Embedding output row count mismatch.")

    return [EmbeddingResult(chunk_id=cid, vector=vec) for cid, vec in zip(ids, matrix)]
