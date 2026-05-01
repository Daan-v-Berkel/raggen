"""Model capability introspection.

Provides ModelInspector.introspect() which loads an embedding model via the
configured backend and returns its runtime capabilities as a plain
ModelCapabilities dataclass.

Backend-specific loading logic lives in backends/onnx.py and backends/torch.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from raggen.core.embeddings.embedder import create_embedder


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded from the given ID or cache."""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelCapabilities:
    """Introspected capabilities of an embedding model."""

    model_id: str
    actual_dim: int          # number of dimensions the model actually outputs
    max_seq_length: int      # maximum tokens the model encodes (incl. special tokens)
    max_batch_size: Optional[int]  # None = no known hard limit (reserved)


# ---------------------------------------------------------------------------
# Inspector
# ---------------------------------------------------------------------------


class ModelInspector:
    """Inspect an embedding model's actual runtime capabilities."""

    @staticmethod
    def introspect(
        model_id: str,
        cache_dir: str | None = None,
        backend: str = "auto",
    ) -> ModelCapabilities:
        """Load the model via the active backend and return its capabilities.

        Raises ModelLoadError if the model cannot be loaded.
        """
        embedder = create_embedder(
            model_id=model_id,
            backend=backend,
            cache_dir=cache_dir,
        )
        return ModelCapabilities(
            model_id=model_id,
            actual_dim=embedder.dim,
            max_seq_length=embedder.max_seq_length,
            max_batch_size=None,
        )
