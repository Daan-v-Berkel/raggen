from __future__ import annotations

from abc import ABC, abstractmethod
from sqlalchemy.engine import Engine


class MetadataBackend(ABC):
    key: str

    @abstractmethod
    def create_schema(self, engine: Engine) -> None:
        """Create all metadata tables in the target database."""

    @abstractmethod
    def drop_schema(self, engine: Engine) -> None:
        """Drop all metadata tables (used for destructive re-init)."""
