from __future__ import annotations

import os
import shutil
from pathlib import Path

from raggen.core.config.project import default_project_config
from raggen.core.results.envelope import ResultEnvelope, ResultMessage
from raggen.core.metadata.store import create_project_state, save_project_state
from raggen.core.metadata.models import ProjectLifecycleState
from raggen.core.results.envelope import init_result


def do_init(
    *,
    root: str = ".",
    force: bool = False,
) -> ResultEnvelope:
    result = init_result("init")

    root_p = Path(root).resolve()
    rag_dir = root_p / ".rag"
    cfg_path = rag_dir / "config.toml"

    if cfg_path.exists() and not force:
        result.errors.append(
            ResultMessage(
                code="PROJECT_ALREADY_INITIALISED",
                message=(
                    "Project already initialised. "
                    "Edit .rag/config.toml or use --force to recreate the scaffold."
                ),
            )
        )
        result.data = {
            "summary": {
                "project_root": str(root_p),
                "config_path": str(cfg_path),
                "force": force,
                "state": "initialised" if rag_dir.exists() else "missing",
            },
            "details": {},
        }
        return result

    if force and rag_dir.exists():
        shutil.rmtree(rag_dir)

    os.makedirs(rag_dir, exist_ok=False)

    cfg = default_project_config(root_p)
    cfg.save(cfg_path)

    state = create_project_state(
        cfg=cfg,
        state=ProjectLifecycleState.INITIALISED,
    )
    state_path = save_project_state(state)

    result.data = {
        "summary": {
            "project_root": str(root_p),
            "config_path": str(cfg_path),
            "force": force,
            "state": state.state.value,
        },
        "details": {
            "project_state_path": str(state_path),
            "storage_backend_key": cfg.storage.backend_key,
            "database_url": cfg.storage.database_url,
            "vector_backend_import": cfg.storage.vector_backend_import,
            "embedding_model": cfg.embedding.model_id,
            "embedding_dim": cfg.embedding.dim,
            "ignore_files": list(cfg.scan.ignore_files),
            "ignore_globs": list(cfg.scan.ignore),
        },
    }
    result.success = True
    return result
