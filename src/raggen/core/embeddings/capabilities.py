"""Model capability introspection.

Provides ``ModelInspector.introspect()`` which loads a sentence-transformers
model and returns its real capabilities (actual output dimension, max sequence
length, etc.) as a plain ``ModelCapabilities`` dataclass.

The private ``_load_sentence_transformer()`` function is the single place in
the codebase that knows how to locate, download, and cache a model.  Both
``ModelInspector`` and ``LocalSentenceTransformerEmbedder.__init__`` call it
so the resolution logic (local path check → disk load or download + save) is
never duplicated.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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
    """Introspected capabilities of a sentence-transformers embedding model."""

    model_id: str
    actual_dim: int          # number of dimensions the model actually outputs
    max_seq_length: int      # maximum tokens the model encodes (incl. special tokens)
    max_batch_size: Optional[int]  # None = no known hard limit (reserved)


# ---------------------------------------------------------------------------
# Shared load helpers (used by ModelInspector and LocalSentenceTransformerEmbedder)
# ---------------------------------------------------------------------------


def _resolve_cache_dir(cache_dir: str | None) -> Path | None:
    if not cache_dir:
        return None
    p = Path(cache_dir)
    return p.expanduser().resolve() if not p.is_absolute() else p


def _local_model_path(cache_dir: Path, model_id: str) -> Path:
    """Returns the expected on-disk path for a model placed inside cache_dir."""
    return cache_dir / model_id


def _is_valid_local_model(path: Path) -> bool:
    return (path / "config.json").exists()


def _load_sentence_transformer(model_id: str, cache_dir: str | None = None):
    """Load a SentenceTransformer model, preferring a local cache when available.

    Resolution order:
    1. If ``cache_dir`` is set and a valid local copy exists there, load from
       disk with ``local_files_only=True`` (no network call).
    2. If ``cache_dir`` is set but no local copy exists, download from the Hub
       and save a copy to ``cache_dir / model_id`` for future use.
    3. If no ``cache_dir`` is given, load normally (HuggingFace cache logic).

    Returns the loaded ``SentenceTransformer`` instance.

    Raises ``ModelLoadError`` (wrapping the original exception) if the model
    cannot be loaded for any reason, so callers get a single, predictable
    exception type regardless of what sentence-transformers raises internally.
    """
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Missing dependency. Install with: pip install sentence-transformers"
        ) from exc

    # Suppress the HuggingFace "Loading weights …" tqdm bar — it is internal
    # implementation noise that clutters CLI output.
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
# Inspector
# ---------------------------------------------------------------------------


class ModelInspector:
    """Inspect an embedding model's actual runtime capabilities."""

    @staticmethod
    def introspect(
        model_id: str,
        cache_dir: str | None = None,
    ) -> ModelCapabilities:
        """Load the model and return its capabilities.

        Uses the same cache resolution logic as ``LocalSentenceTransformerEmbedder``
        so the two are always consistent.

        Raises ``ModelLoadError`` if the model cannot be loaded.
        """
        model = _load_sentence_transformer(model_id, cache_dir)
        return ModelCapabilities(
            model_id=model_id,
            actual_dim=int(model.get_sentence_embedding_dimension()),
            max_seq_length=int(model.max_seq_length),
            max_batch_size=None,
        )
