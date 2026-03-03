from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import tomllib as _toml_reader
except Exception:
    _toml_reader = None

try:
    import tomli_w as _toml_writer
except Exception:
    _toml_writer = None


@dataclass
class ScanConfig:
    use_gitignore: bool = True
    ignore: List[str] = field(default_factory=lambda: [".venv/", "node_modules/"])


@dataclass
class ChunkingConfig:
    strategy: str = "fixed"
    unit: str = "chars"
    chunk_size: int = 1200
    overlap: int = 200


@dataclass
class EmbeddingConfig:
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim: int = 384
    normalize: bool = True
    batch_size: int = 32


@dataclass
class StorageConfig:
    backend_key: str = "sqlite_vec"
    database_url: str = "sqlite:///./.rag/rag.db"
    vector_backend_import: str = "raggen.core.store.vector_backends.sqlite_vec:SQLiteVecBackend"
    destructive_default: bool = False


@dataclass
class QueryConfig:
    model_id: str = ""


@dataclass
class RerankConfig:
    enabled: bool = False
    model_id: str = ""
    top_n: int = 50


@dataclass
class ProjectConfig:
    project_root: Path = Path('.')
    schema_version: str = "v1"
    scan: ScanConfig = field(default_factory=ScanConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)


def default_project_config(root: Path | str) -> ProjectConfig:
    return ProjectConfig(project_root=Path(root))


def load_project_config(path: Path) -> ProjectConfig:
    if not _toml_reader:
        raise RuntimeError("tomllib not available; cannot read TOML config")
    with open(path, "rb") as fh:
        data = _toml_reader.load(fh)
    pc = ProjectConfig()
    project = data.get("project", {})
    pc.project_root = Path(project.get("root", "."))
    pc.schema_version = project.get("schema_version", "v1")
    scan = data.get("scan", {})
    pc.scan.use_gitignore = scan.get("use_gitignore", True)
    pc.scan.ignore = scan.get("ignore", pc.scan.ignore)
    chunk = data.get("chunking", {})
    pc.chunking.strategy = chunk.get("strategy", pc.chunking.strategy)
    pc.chunking.unit = chunk.get("unit", pc.chunking.unit)
    pc.chunking.chunk_size = int(chunk.get("chunk_size", pc.chunking.chunk_size))
    pc.chunking.overlap = int(chunk.get("overlap", pc.chunking.overlap))
    emb = data.get("embedding", {})
    pc.embedding.model_id = emb.get("model_id", pc.embedding.model_id)
    pc.embedding.dim = int(emb.get("dim", pc.embedding.dim))
    pc.embedding.normalize = bool(emb.get("normalize", pc.embedding.normalize))
    pc.embedding.batch_size = int(emb.get("batch_size", pc.embedding.batch_size))
    stor = data.get("storage", {})
    pc.storage.backend_key = stor.get("backend_key", pc.storage.backend_key)
    pc.storage.database_url = stor.get("database_url", pc.storage.database_url)
    pc.storage.vector_backend_import = stor.get("vector_backend_import", pc.storage.vector_backend_import)
    pc.storage.destructive_default = bool(stor.get("destructive_default", pc.storage.destructive_default))
    query = data.get("query", {})
    pc.query.model_id = query.get("model_id", "")
    rer = data.get("rerank", {})
    pc.rerank.enabled = bool(rer.get("enabled", False))
    pc.rerank.model_id = rer.get("model_id", "")
    pc.rerank.top_n = int(rer.get("top_n", pc.rerank.top_n))
    return pc


def save_project_config(cfg: ProjectConfig, path: Path) -> None:
    # minimal TOML writer if tomli_w not available
    d = {
        "project": {"root": str(cfg.project_root), "schema_version": cfg.schema_version},
        "scan": {"use_gitignore": cfg.scan.use_gitignore, "ignore": cfg.scan.ignore},
        "chunking": {"strategy": cfg.chunking.strategy, "unit": cfg.chunking.unit, "chunk_size": cfg.chunking.chunk_size, "overlap": cfg.chunking.overlap},
        "embedding": {"model_id": cfg.embedding.model_id, "dim": cfg.embedding.dim, "normalize": cfg.embedding.normalize, "batch_size": cfg.embedding.batch_size},
        "storage": {"backend_key": cfg.storage.backend_key, "database_url": cfg.storage.database_url, "vector_backend_import": cfg.storage.vector_backend_import, "destructive_default": cfg.storage.destructive_default},
        "query": {"model_id": cfg.query.model_id},
        "rerank": {"enabled": cfg.rerank.enabled, "model_id": cfg.rerank.model_id, "top_n": cfg.rerank.top_n},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if _toml_writer:
        with open(path, "wb") as fh:
            fh.write(_toml_writer.dumps(d).encode("utf-8"))
    else:
        # naive writer
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("[project]\n")
            fh.write(f"root = \"{d['project']['root']}\"\n")
            fh.write(f"schema_version = \"{d['project']['schema_version']}\"\n\n")
            fh.write("[scan]\n")
            fh.write(f"use_gitignore = {str(d['scan']['use_gitignore']).lower()}\n")
            fh.write("ignore = [\"" + "\", \"".join(d['scan']['ignore']) + "\"]\n\n")
            fh.write("[chunking]\n")
            fh.write(f"strategy = \"{d['chunking']['strategy']}\"\n")
            fh.write(f"unit = \"{d['chunking']['unit']}\"\n")
            fh.write(f"chunk_size = {d['chunking']['chunk_size']}\n")
            fh.write(f"overlap = {d['chunking']['overlap']}\n\n")
            fh.write("[embedding]\n")
            fh.write(f"model_id = \"{d['embedding']['model_id']}\"\n")
            fh.write(f"dim = {d['embedding']['dim']}\n")
            fh.write(f"normalize = {str(d['embedding']['normalize']).lower()}\n")
            fh.write(f"batch_size = {d['embedding']['batch_size']}\n\n")
            fh.write("[storage]\n")
            fh.write(f"backend_key = \"{d['storage']['backend_key']}\"\n")
            fh.write(f"database_url = \"{d['storage']['database_url']}\"\n")
            fh.write(f"vector_backend_import = \"{d['storage']['vector_backend_import']}\"\n")
            fh.write(f"destructive_default = {str(d['storage']['destructive_default']).lower()}\n\n")
            fh.write("[query]\n")
            fh.write(f"model_id = \"{d['query']['model_id']}\"\n\n")
            fh.write("[rerank]\n")
            fh.write(f"enabled = {str(d['rerank']['enabled']).lower()}\n")
            fh.write(f"model_id = \"{d['rerank']['model_id']}\"\n")
            fh.write(f"top_n = {d['rerank']['top_n']}\n")
