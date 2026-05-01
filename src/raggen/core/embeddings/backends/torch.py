from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from raggen.core.embeddings.capabilities import ModelLoadError


# ---------------------------------------------------------------------------
# Cache helpers (torch / HuggingFace checkpoint layout)
# ---------------------------------------------------------------------------


def _resolve_cache_dir(cache_dir: str | None) -> Path | None:
    if not cache_dir:
        return None
    p = Path(cache_dir)
    return p.expanduser().resolve() if not p.is_absolute() else p


def _local_model_path(cache_dir: Path, model_id: str) -> Path:
    return cache_dir / model_id


def _is_valid_local_model(path: Path) -> bool:
    """A HuggingFace checkpoint is considered valid when config.json is present."""
    return (path / "config.json").exists()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


def _load_model(model_id: str, cache_dir: str | None = None):
    """Load a SentenceTransformer model, preferring a local cache when available.

    Resolution order:
    1. cache_dir set + valid local copy → load from disk, no network call.
    2. cache_dir set + no local copy   → download from Hub, save to cache.
    3. No cache_dir                    → default HuggingFace cache logic.

    Raises ModelLoadError on any load failure.
    """
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is not installed.\n"
            "  Install with: pip install 'raggen[torch]'"
        ) from exc

    try:
        import transformers  # noqa: PLC0415
        transformers.logging.set_verbosity_error()
    except Exception:  # noqa: BLE001
        pass

    resolved = _resolve_cache_dir(cache_dir)

    try:
        if resolved is not None:
            local_path = _local_model_path(resolved, model_id)
            if _is_valid_local_model(local_path):
                return SentenceTransformer(
                    str(local_path), device="cpu", local_files_only=True
                )
            model = SentenceTransformer(model_id, device="cpu")
            local_path.mkdir(parents=True, exist_ok=True)
            model.save(str(local_path))
            return model
        return SentenceTransformer(model_id, device="cpu")

    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load embedding model '{model_id}'.\n"
            "Check that the model ID is correct and that the cache directory "
            "is accessible.\n\n"
            f"Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class TorchEmbedder:
    """Embedding backend backed by sentence-transformers (PyTorch).

    Install the torch backend with: pip install 'raggen[torch]'

    Determinism: embeddings are stable for a fixed model version + input text.
    Changing model_id may change output dimensions — store model_id alongside
    your vectors.
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
        self.normalize = normalize
        self._model = _load_model(model_id, cache_dir)

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    @property
    def max_seq_length(self) -> int:
        return int(self._model.max_seq_length)

    def get_length_function(self) -> Callable[[str], int]:
        tokenizer = self._model.tokenizer

        def _count_tokens(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))

        return _count_tokens

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        normalize: bool | None = None,
    ) -> np.ndarray:
        """Returns a 2-D float32 array of shape (len(texts), dim)."""
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size if batch_size is not None else self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize if normalize is not None else self.normalize,
        )
        return np.asarray(vectors, dtype=np.float32)
