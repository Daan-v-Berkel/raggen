"""Per-model capability cache stored under .rag/metadata/model_specs/.

``ModelSpecsCache`` is the bridge between build-time model introspection and
runtime commands (ingest, query) that need model capabilities without loading
the model again.

Write path (build only):
    cache = ModelSpecsCache(root / ".rag" / "metadata" / "model_specs")
    caps  = ModelInspector.introspect(model_id, cache_dir)   # or from cache
    cache.put(caps)

Read path (ingest / query):
    cache = ModelSpecsCache(root / ".rag" / "metadata" / "model_specs")
    caps  = cache.get(model_id)
    if caps is None:
        raise MissingModelSpecsError(model_id)
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from raggen.core.embeddings.capabilities import ModelCapabilities


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MissingModelSpecsError(RuntimeError):
    """Raised when a model's capability spec is not found in the cache.

    This should never occur in normal usage because ``rag build`` always writes
    the spec before ``rag ingest`` or ``rag query`` are run.  It protects
    against manually deleted cache files or running ingest/query on a project
    that has never been built.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(
            f"Model capabilities not cached for '{model_id}'.\n"
            "Run 'rag build' before ingesting or querying."
        )


# ---------------------------------------------------------------------------
# File-name sanitisation
# ---------------------------------------------------------------------------

_SAFE_CHARS = re.compile(r"[^a-zA-Z0-9\-.]")


def _model_id_to_filename(model_id: str) -> str:
    """Convert a model ID to a safe, human-readable filename (no extension).

    Rules:
    - ``/`` → ``__``  (namespace separator, common in HuggingFace IDs)
    - Any other character that is not alphanumeric, ``-``, or ``.`` → ``_``

    Examples:
        "sentence-transformers/all-MiniLM-L6-v2" → "sentence-transformers__all-MiniLM-L6-v2"
        "BAAI/bge-small-en-v1.5"                 → "BAAI__bge-small-en-v1.5"
    """
    safe = model_id.replace("/", "__")
    return _SAFE_CHARS.sub("_", safe)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class ModelSpecsCache:
    """Read/write per-model capability files from a directory on disk.

    Each model gets its own file keyed by model ID.  Files are intentionally
    left without a ``.json`` extension to discourage manual editing.
    """

    def __init__(self, specs_dir: Path) -> None:
        self._dir = specs_dir

    def _path_for(self, model_id: str) -> Path:
        return self._dir / _model_id_to_filename(model_id)

    def exists(self, model_id: str) -> bool:
        """Return True if a cached spec file exists for ``model_id``."""
        return self._path_for(model_id).exists()

    def get(self, model_id: str) -> Optional[ModelCapabilities]:
        """Return cached capabilities, or ``None`` on a cache miss."""
        path = self._path_for(model_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ModelCapabilities(
            model_id=payload["model_id"],
            actual_dim=payload["actual_dim"],
            max_seq_length=payload["max_seq_length"],
            max_batch_size=payload.get("max_batch_size"),
        )

    def put(self, caps: ModelCapabilities) -> None:
        """Write capabilities to cache, overwriting any existing entry."""
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path_for(caps.model_id)
        payload = {
            "model_id": caps.model_id,
            "actual_dim": caps.actual_dim,
            "max_seq_length": caps.max_seq_length,
            "max_batch_size": caps.max_batch_size,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(
            json.dumps(payload, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )
