from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np


def _key(chunk_id: str, model_id: str) -> str:
    raw = f"{model_id}::{chunk_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class NpyDirCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, chunk_id: str, model_id: str):
        p = self.root / f"{_key(chunk_id, model_id)}.npy"
        if not p.exists():
            return None
        return np.load(p)

    def put(self, chunk_id: str, model_id: str, vector: np.ndarray):
        p = self.root / f"{_key(chunk_id, model_id)}.npy"
        np.save(p, np.asarray(vector, dtype=np.float32))
