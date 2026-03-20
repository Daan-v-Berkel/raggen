from pydantic import BaseModel
from enum import Enum


class ProjectLifecycleState(str, Enum):
    INITIALISED = "initialised"
    SET_UP = "set_up"


class FoundationalConfigSnapshot(BaseModel):
    project_root: str
    schema_version: str

    embedding_model: str
    embedding_dim: int

    storage_backend_key: str
    database_url: str
    vector_backend_import: str


class ProjectState(BaseModel):
    state: ProjectLifecycleState
    updated_at: str
    foundation: FoundationalConfigSnapshot
