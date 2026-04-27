from __future__ import annotations

from dataclasses import dataclass, field
from fancy_dataclass import ConfigDataclass, TOMLDataclass
from pathlib import Path
from typing import List
import json


@dataclass
class ScanConfig(TOMLDataclass):
    ignore_files: List[str] = field(default_factory=lambda: [".gitignore"])
    ignore: List[str] = field(default_factory=lambda: [".venv/", "node_modules/"])
    max_encoding_error_ratio: float = 0.05


@dataclass
class GroupChunkingConfig(TOMLDataclass):
    strategy: str = "fixed"
    unit: str = "chars"
    chunk_size: int = 1200
    overlap: int = 200


@dataclass
class FileGroupConfig(TOMLDataclass):
    extensions: List[str] = field(default_factory=list)


def _default_file_groups():
    return {
        "fallback": FileGroupConfig(extensions=[]),
    }


def _default_chunking():
    return {
        "fallback": GroupChunkingConfig(
            strategy="fixed",
            unit="chars",
            chunk_size=1000,
            overlap=100,
        ),
    }


@dataclass
class EmbeddingConfig(TOMLDataclass):
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_cache_dir: str = ".rag/models/"
    dim: int = 384
    normalize: bool = True
    batch_size: int = 32


@dataclass
class StorageConfig(TOMLDataclass):
    backend_key: str = "sqlite_vec"
    database_url: str = "sqlite:///./.rag/rag.db"
    # Leave empty to use the built-in backend for backend_key.
    # Set to a 'module:ClassName' import path for a custom backend plugin.
    vector_backend_import: str = ""
    destructive_default: bool = False


@dataclass
class QueryConfig(TOMLDataclass):
    model_id: str = ""
    top_k: int = 8


@dataclass
class RerankConfig(TOMLDataclass):
    enabled: bool = False
    model_id: str = ""
    top_n: int = 50


@dataclass
class GenerationConfig(TOMLDataclass):
    enabled: bool = False
    provider: str = ""
    model_id: str = ""


@dataclass
class ProjectConfig(ConfigDataclass, TOMLDataclass):
    project_root: Path = Path(".")
    schema_version: str = "v1"
    scan: ScanConfig = field(default_factory=ScanConfig)
    file_groups: dict[str, FileGroupConfig] = field(
        default_factory=_default_file_groups
    )
    chunking: dict[str, GroupChunkingConfig] = field(default_factory=_default_chunking)
    fallback_group: str = "fallback"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    notes: List[str] = field(default_factory=lambda: [])

    def to_row(self) -> dict:
        """
        A simplified version of the configuration as_dict, for inserting into the database
        """
        return {
            "schema_version": self.schema_version,
            "backend_key": self.storage.backend_key,
            "database_url": self.storage.database_url,
            "embedding_model_id": self.embedding.model_id,
            "embedding_dim": self.embedding.dim,
            "embedding_normalized": self.embedding.normalize,
            "query_model_id": self.query.model_id,
            "notes_json": json.dumps(self.notes),
        }


def default_project_config(root: Path | str) -> ProjectConfig:
    return ProjectConfig(project_root=Path(root))
