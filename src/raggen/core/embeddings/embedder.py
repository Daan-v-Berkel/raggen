from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import numpy as np

from raggen.core.embeddings.capabilities import _load_sentence_transformer


@dataclass(frozen=True)
class EmbeddingResult:
    chunk_id: str
    vector: np.ndarray  # shape: (dim,)


class LocalSentenceTransformerEmbedder:
    """
    Minimal CPU embedder using sentence-transformers.

    Model loading is deferred to the first instantiation (lazy import) via the
    shared ``_load_sentence_transformer`` helper in ``capabilities.py``.  That
    helper also handles local-cache lookup, HuggingFace download + save, and
    verbosity suppression — nothing is duplicated here.

    Determinism notes:
    - Embeddings are stable for a given model version + text.
    - If you change model_id, dimension may change — store model_id with vectors.
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
        self._model = _load_sentence_transformer(model_id, cache_dir)

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    @property
    def max_seq_length(self) -> int:
        """Maximum number of tokens the model can encode (including special tokens)."""
        return int(self._model.max_seq_length)

    def get_length_function(self):
        """Return a callable (str) -> int that counts tokens using this model's tokenizer.

        Special tokens (CLS, SEP) are excluded so that ``chunk_size`` in the
        config maps directly to content tokens, not the model's internal overhead.
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

    ``chunks`` must have:
      - chunk.chunk_id (str)
      - chunk.text (str)
    """
    texts = [getattr(ch, "text") for ch in chunks]
    ids = [getattr(ch, "chunk_id") for ch in chunks]

    matrix = embedder.embed_texts(texts)

    if matrix.shape[0] != len(ids):
        raise RuntimeError("Embedding output row count mismatch.")

    return [EmbeddingResult(chunk_id=cid, vector=vec) for cid, vec in zip(ids, matrix)]
