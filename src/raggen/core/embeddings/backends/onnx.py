from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from raggen.core.embeddings.capabilities import ModelLoadError

# Attribute paths where fastembed exposes its inner ONNX model, ordered by
# version prevalence. fastembed renamed _model → model at some point.
_INNER_MODEL_ATTRS = ("model", "_model")


def _extract_tokenizer(fe_model, model_id: str):
    """Return the tokenizers.Tokenizer buried inside a fastembed TextEmbedding."""
    for attr in _INNER_MODEL_ATTRS:
        inner = getattr(fe_model, attr, None)
        if inner is not None and hasattr(inner, "tokenizer"):
            return inner.tokenizer
    raise ModelLoadError(
        f"Could not access tokenizer for model '{model_id}'. "
        "fastembed's internal API may have changed. "
        f"Searched attributes: {_INNER_MODEL_ATTRS}"
    )


def _extract_max_length(fe_model, tokenizer) -> int:
    """Return the model's token capacity, falling back to 512 if not found."""
    for attr in _INNER_MODEL_ATTRS:
        inner = getattr(fe_model, attr, None)
        if inner is not None and hasattr(inner, "max_length"):
            return int(inner.max_length)
    return int(getattr(tokenizer, "model_max_length", 512))


class OnnxEmbedder:
    """Embedding backend backed by fastembed (ONNX Runtime).

    Default backend — no GPU driver or CUDA install required.

    Note: the normalize parameter is stored for interface compatibility but has
    no effect — fastembed always L2-normalises output vectors. If you need
    un-normalised embeddings, use the torch backend instead.

    Determinism: ONNX quantised models produce vectors that are numerically
    close (< 0.001 cosine error) but not bit-identical to the equivalent
    PyTorch model. Existing databases built with the torch backend should be
    re-ingested after switching to this backend.
    """

    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str | None = None,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        self.normalize = normalize  # stored for interface compatibility; no effect

        try:
            from fastembed import TextEmbedding  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "fastembed is not installed.\n"
                "  Install with: pip install raggen"
            ) from exc

        try:
            kwargs: dict = {"model_name": model_id}
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
            self._model = TextEmbedding(**kwargs)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load embedding model '{model_id}'.\n"
                "Check that the model ID is in fastembed's supported model list "
                "and that the cache directory is accessible.\n"
                "For models not supported by fastembed, use the torch backend: "
                "set backend = 'torch' in .rag/config.toml and install "
                "pip install 'raggen[torch]'\n\n"
                f"Original error: {exc}"
            ) from exc

        # Probe a single string to determine the output dimension. This happens
        # once at construction time when the model is already loaded.
        probe = np.array(list(self._model.embed(["probe"], batch_size=1)))
        self._dim = int(probe.shape[1])

        # fastembed's internal structure has changed across versions. Try known
        # attribute paths in order rather than hardcoding one.
        self._tokenizer = _extract_tokenizer(self._model, model_id)
        self._max_seq_length = _extract_max_length(self._model, self._tokenizer)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    def get_length_function(self) -> Callable[[str], int]:
        tokenizer = self._tokenizer

        def _count_tokens(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False).ids)

        return _count_tokens

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        normalize: bool | None = None,  # no effect; fastembed always normalises
    ) -> np.ndarray:
        """Returns a 2-D float32 array of shape (len(texts), dim)."""
        effective_batch = batch_size if batch_size is not None else self.batch_size
        vectors = np.array(
            list(self._model.embed(list(texts), batch_size=effective_batch))
        )
        return vectors.astype(np.float32)
