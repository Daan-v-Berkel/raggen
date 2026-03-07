from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple
from sqlalchemy.engine import Engine


class VectorBackend(ABC):
    key: str

    @abstractmethod
    def supports(self, engine: Engine) -> bool:
        """Return True if this backend can operate with the given engine/dialect."""

    @abstractmethod
    def create_schema(self, engine: Engine, dim: int) -> None:
        """Create vector storage schema for the configured dimension."""

    @abstractmethod
    def drop_schema(self, engine: Engine) -> None:
        """Drop vector storage schema (used for destructive re-init)."""

    @abstractmethod
    def upsert_vectors(
        self,
        engine: Engine,
        *,
        vectors: List[Tuple[str, List[float]]],
        embedding_model_id: str,
        dim: int,
        normalized: bool,
    ) -> None:
        """Insert/replace vectors keyed by chunk_id."""

    @abstractmethod
    def delete_vectors(self, engine: Engine, *, chunks: List[str]) -> None:
        """Delete vectors by chunk_id"""

    @abstractmethod
    def search(self, engine: Engine, *, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        """the main entry point for querying the database.
        Return [(chunk_id, score)] where score is backend-native distance.
        Lower is better.
        """
