from __future__ import annotations

from typing import Callable, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class EmbedderBackend(Protocol):
    """Structural interface satisfied by all embedding backend implementations.

    Both OnnxEmbedder and TorchEmbedder conform to this protocol, as do test
    doubles — no inheritance required.
    """

    model_id: str
    batch_size: int
    normalize: bool

    @property
    def dim(self) -> int:
        """Number of dimensions in the output embedding vectors."""
        ...

    @property
    def max_seq_length(self) -> int:
        """Maximum tokens the model encodes, including special tokens."""
        ...

    def get_length_function(self) -> Callable[[str], int]:
        """Return a token-counting callable for this model's tokenizer.

        The returned function excludes special tokens so that chunk_size in the
        config maps directly to content tokens.
        """
        ...

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        normalize: bool | None = None,
    ) -> np.ndarray:
        """Embed a sequence of strings.

        Returns a 2-D float32 array of shape (len(texts), dim).
        """
        ...
