from __future__ import annotations

from typing import Optional

from pydantic import BaseModel
from enum import Enum


class ProjectLifecycleState(str, Enum):
    INITIALISED = "initialised"
    SET_UP = "set_up"


class FoundationalConfigSnapshot(BaseModel):
    project_root: str
    schema_version: str

    embedding_model: str
    # None before the first build resolves the dim from the model specs cache.
    # After a successful build this is always a concrete integer.
    embedding_dim: Optional[int] = None

    storage_backend_key: str
    database_url: str
    vector_backend_import: str

    # SHA-256 hashes of boundary-affecting chunking fields, keyed by group name.
    # Empty dict for projects built before this feature was introduced — those
    # produce no drift warnings at ingest time.
    chunking_hashes: dict[str, str] = {}


class ProjectState(BaseModel):
    state: ProjectLifecycleState
    updated_at: str
    foundation: FoundationalConfigSnapshot
