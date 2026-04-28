from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "Missing dependency. Install with: pip install sentence-transformers"
    ) from e


@dataclass(frozen=True)
class EmbeddingResult:
    chunk_id: str
    vector: np.ndarray  # shape: (dim,)


def _resolve_cache_dir(cache_dir: str | None) -> Path | None:
    if not cache_dir:
        return None
    p = Path(cache_dir)
    return p.expanduser().resolve() if not p.is_absolute() else p


def _local_model_path(cache_dir: Path, model_id: str) -> Path:
    """Returns the expected on-disk path for a manually-placed model inside cache_dir."""
    return cache_dir / model_id


def _is_valid_local_model(path: Path) -> bool:
    return (path / "config.json").exists()



class LocalSentenceTransformerEmbedder:
    """
    Minimal CPU embedder using sentence-transformers.

    Determinism notes:
    - Embeddings should be stable for a given model version + text.
    - If you change model_id, dimension may change, so store model_id with vectors.
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
        resolved = _resolve_cache_dir(cache_dir)

        if resolved is not None:
            local_path = _local_model_path(resolved, model_id)
            if _is_valid_local_model(local_path):
                self._model = SentenceTransformer(
                    str(local_path), device="cpu", local_files_only=True
                )
            else:
                self._model = SentenceTransformer(model_id, device="cpu")
                local_path.mkdir(parents=True, exist_ok=True)
                self._model.save(str(local_path))
        else:
            self._model = SentenceTransformer(model_id, device="cpu")

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def get_length_function(self):
        """Return a callable (str) -> int that counts tokens using this model's tokenizer.

        The tokenizer is already downloaded as part of the model — no extra
        network calls or dependencies are needed.  Special tokens (CLS, SEP)
        are excluded from the count so that ``chunk_size`` in the config maps
        directly to content tokens, not the model's internal overhead.
        """
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
        """Returns a 2D array: shape (n, dim), dtype float32."""
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size if batch_size is not None else self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize if normalize is not None else self.normalize,
        )
        return np.asarray(vectors, dtype=np.float32)


def embed_chunks(
    embedder: LocalSentenceTransformerEmbedder,
    chunks: Sequence,
) -> List[EmbeddingResult]:
    """
    Embed a sequence of chunks.

    `chunks` must have:
      - chunk.chunk_id (str)
      - chunk.text (str)
    """
    texts = [getattr(ch, "text") for ch in chunks]
    ids = [getattr(ch, "chunk_id") for ch in chunks]

    matrix = embedder.embed_texts(texts)

    if matrix.shape[0] != len(ids):
        raise RuntimeError("Embedding output row count mismatch.")

    return [EmbeddingResult(chunk_id=cid, vector=vec) for cid, vec in zip(ids, matrix)]
