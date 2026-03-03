from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import json
import hashlib


@dataclass
class RagInitConfig:
    schema_version: str = "v1"
    backend_key: str = "sqlite_vec"
    database_url: str = "sqlite:///./.rag/rag.db"
    embedding_model_id: str = ""
    embedding_dim: int = 0
    embedding_normalized: bool = True
    query_model_id: Optional[str] = None
    chunk_config_hash: str = ""
    notes: Dict[str, Any] = field(default_factory=dict)
    vector_backend_import: str = ""

    def to_row(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        return {
            "id": 1,
            "created_at": now,
            "schema_version": self.schema_version,
            "backend_key": self.backend_key,
            "database_url": self.database_url,
            "embedding_model_id": self.embedding_model_id,
            "embedding_dim": int(self.embedding_dim),
            "embedding_normalized": 1 if self.embedding_normalized else 0,
            "query_model_id": self.query_model_id,
            "chunk_config_hash": self.chunk_config_hash,
            "notes_json": json.dumps(self.notes, sort_keys=True, ensure_ascii=False),
        }

    def stable_fingerprint(self) -> str:
        # Create a canonical JSON of key fields and notes
        payload = {
            "schema_version": self.schema_version,
            "backend_key": self.backend_key,
            "database_url": self.database_url,
            "embedding_model_id": self.embedding_model_id,
            "embedding_dim": int(self.embedding_dim),
            "embedding_normalized": bool(self.embedding_normalized),
            "query_model_id": self.query_model_id,
            "chunk_config_hash": self.chunk_config_hash,
            "notes": self.notes,
        }
        js = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(js.encode("utf-8")).hexdigest()
