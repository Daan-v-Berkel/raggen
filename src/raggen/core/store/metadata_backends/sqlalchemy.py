from __future__ import annotations

from sqlalchemy.engine import Engine

from .base import MetadataBackend
from raggen.core.store.metadata_schema import metadata


class SqlalchemyMetadataBackend(MetadataBackend):
    key = "sqlalchemy"

    def create_schema(self, engine: Engine) -> None:
        metadata.create_all(engine)

    def drop_schema(self, engine: Engine) -> None:
        metadata.drop_all(engine)
