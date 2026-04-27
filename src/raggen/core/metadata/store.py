from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from raggen.core.config.project import ProjectConfig
from raggen.core.metadata.models import (
    FoundationalConfigSnapshot,
    ProjectState,
)
from raggen.core.store.plugin_loader import resolve_vector_backend_import


def metadata_dir(project_root: str | Path) -> Path:
    return Path(project_root).resolve() / ".rag" / "metadata"


def project_state_path(project_root: str | Path) -> Path:
    return metadata_dir(project_root) / "project_state.json"


def snapshot_foundational_config(cfg: ProjectConfig) -> FoundationalConfigSnapshot:
    return FoundationalConfigSnapshot(
        project_root=str(Path(cfg.project_root).resolve()),
        schema_version=cfg.schema_version,
        embedding_model=cfg.embedding.model_id,
        embedding_dim=cfg.embedding.dim,
        storage_backend_key=cfg.storage.backend_key,
        database_url=cfg.storage.database_url,
        vector_backend_import=resolve_vector_backend_import(
            cfg.storage.backend_key, cfg.storage.vector_backend_import
        ),
    )


def load_project_state(project_root: str | Path) -> ProjectState | None:
    path = project_state_path(project_root)
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    return ProjectState.model_validate(payload)


def save_project_state(state: ProjectState) -> Path:
    path = project_state_path(state.foundation.project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            state.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def create_project_state(
    *,
    cfg: ProjectConfig,
    state: str,
) -> ProjectState:
    return ProjectState(
        state=state,
        updated_at=datetime.now(timezone.utc).isoformat(),
        foundation=snapshot_foundational_config(cfg),
    )
