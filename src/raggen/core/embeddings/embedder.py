from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional, Dict, Any
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


class LocalSentenceTransformerEmbedder:
    """
    Minimal CPU embedder using sentence-transformers.

    Determinism notes:
    - Embeddings should be stable for a given model version + text.
    - If you change model_id, dimension may change, so store model_id with vectors.
    """

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_id = model_id
        # device="cpu" is explicit; normalize_embeddings helps cosine similarity later.
        self._model = SentenceTransformer(model_id, device="cpu")

    @property
    def dim(self) -> int:
        # sentence-transformers can give this
        return int(self._model.get_sentence_embedding_dimension())

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Returns a 2D array: shape (n, dim), dtype float32.
        """
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
        )
        # ensure numpy float32
        vectors = np.asarray(vectors, dtype=np.float32)
        return vectors


def embed_chunks(
    embedder: LocalSentenceTransformerEmbedder,
    chunks: Sequence[Any],
    *,
    batch_size: int = 32,
    normalize: bool = True,
    cache_get: Optional[callable] = None,
    cache_put: Optional[callable] = None,
) -> List[EmbeddingResult]:
    """
    Minimal chunk embedding function.

    `chunks` must have:
      - chunk.chunk_id (str)
      - chunk.text (str)

    Optional caching:
      - cache_get(chunk_id, model_id) -> np.ndarray | None
      - cache_put(chunk_id, model_id, vector: np.ndarray) -> None
    """
    results: List[EmbeddingResult] = []

    # 1) Split into cached vs missing
    missing_ids: List[str] = []
    missing_texts: List[str] = []

    for ch in chunks:
        cid = getattr(ch, "chunk_id")
        txt = getattr(ch, "text")

        if cache_get is not None:
            cached = cache_get(cid, embedder.model_id)
            if cached is not None:
                vec = np.asarray(cached, dtype=np.float32)
                if vec.ndim != 1:
                    raise ValueError(f"Cached vector for {cid} is not 1D.")
                results.append(EmbeddingResult(chunk_id=cid, vector=vec))
                continue

        missing_ids.append(cid)
        missing_texts.append(txt)

    # 2) Embed missing in batches
    if missing_texts:
        matrix = embedder.embed_texts(
            missing_texts, batch_size=batch_size, normalize=normalize)
        if matrix.shape[0] != len(missing_ids):
            raise RuntimeError("Embedding output row count mismatch.")

        for cid, vec in zip(missing_ids, matrix):
            # vec is 1D (dim,)
            if cache_put is not None:
                cache_put(cid, embedder.model_id, vec)
            results.append(EmbeddingResult(chunk_id=cid, vector=vec))

    # 3) Return in same order as input chunks (important for pipeline predictability)
    order = {getattr(ch, "chunk_id"): i for i, ch in enumerate(chunks)}
    results.sort(key=lambda r: order[r.chunk_id])
    return results
