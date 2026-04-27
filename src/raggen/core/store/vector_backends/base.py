from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple
from sqlalchemy.engine import Connection, Engine


class VectorBackend(ABC):
    """Abstract base class for vector storage backends.

    Implement this class to add support for a new vector store. Register
    your implementation by pointing `vector_backend_import` in the project's
    `[storage]` config to the fully-qualified class path, e.g.:

        [storage]
        backend_key = "my_backend"
        vector_backend_import = "my_package.backends:MyVectorBackend"

    ## Connection vs Engine

    The write methods (`upsert_vectors`, `delete_vectors`) receive a
    SQLAlchemy `Connection` from the caller. This is intentional: ingest
    wraps document storage, chunk storage, embedding metadata, and vector
    storage in a single transaction. Accepting the caller's connection lets
    your backend participate in that transaction so a failure rolls everything
    back atomically.

    `search` receives an `Engine` instead and opens its own connection
    internally. Retrieval is read-only and never needs to share a write
    transaction, so owning its own connection is simpler and correct.

    ## Score convention

    `search` must return scores where **lower is better** (i.e. distance,
    not similarity). Both built-in backends use cosine distance. If your
    backend returns similarity scores, invert them before returning.
    """

    key: str

    @abstractmethod
    def supports(self, engine: Engine) -> bool:
        """Return True if this backend can operate with the given SQLAlchemy engine.

        Called during `rag build` to verify the configured database URL is
        compatible with this backend before any schema is created.
        """

    @abstractmethod
    def create_schema(self, engine: Engine, dim: int) -> None:
        """Create the vector storage schema for embedding dimension *dim*.

        Called once during `rag build`. Use `IF NOT EXISTS` guards so the
        method is safe to call on a database that already has the schema.
        """

    @abstractmethod
    def drop_schema(self, engine: Engine) -> None:
        """Drop the vector storage schema.

        Called during `rag build --destructive`. Should be a no-op if the
        schema does not exist.
        """

    @abstractmethod
    def upsert_vectors(
        self,
        conn: Connection,
        *,
        vectors: List[Tuple[str, List[float]]],
        embedding_model_id: str,
        dim: int,
        normalized: bool,
    ) -> None:
        """Insert or replace vectors keyed by chunk_id.

        *conn* is the caller's active transactional connection — execute all
        statements against it so this operation is part of the surrounding
        ingest transaction.

        *vectors* is a list of (chunk_id, embedding) pairs where each
        embedding is a flat list of *dim* floats.
        """

    @abstractmethod
    def delete_vectors(self, conn: Connection, *, chunks: List[str]) -> None:
        """Delete vectors for the given chunk_ids.

        *conn* is the caller's active transactional connection — execute all
        statements against it so this operation is part of the surrounding
        ingest transaction.
        """

    @abstractmethod
    def search(
        self,
        engine: Engine,
        *,
        query_vector: list[float],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Return the *top_k* nearest vectors to *query_vector*.

        Opens its own connection via *engine* — do not expect a surrounding
        transaction.

        Returns a list of (chunk_id, score) tuples ordered by ascending score
        (lower = more similar). Return at most *top_k* results.
        """
