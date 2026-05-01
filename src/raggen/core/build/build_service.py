from __future__ import annotations

import pathlib
from raggen.core.config.project import ProjectConfig
from raggen.core.embeddings.capabilities import ModelInspector
from raggen.core.embeddings.model_specs_cache import ModelSpecsCache
from raggen.core.validation.project_validator import ProjectValidator
from raggen.core.runtime import get_engine as _get_engine
from raggen.core.metadata.compare import changed_foundation_fields, classify_foundation_changes
from raggen.core.metadata.models import ProjectLifecycleState
from raggen.core.metadata.store import (
    create_project_state,
    load_project_state,
    project_state_path,
    snapshot_foundational_config,
    save_project_state,
)
from raggen.core.results import ResultEnvelope, ResultMessage, init_result
from raggen.core.runs import get_run_store, persist_result
from raggen.core.store.initializer import init_database


@persist_result(get_run_store)
def do_build(
    *,
    config_path: str | None = None,
    destructive: bool = False,
) -> ResultEnvelope:
    result = init_result("build")

    cfg = ProjectConfig.get_config()
    root_p = cfg.project_root
    cfg_path = pathlib.Path.joinpath(root_p, config_path)

    state_path = project_state_path(root_p)

    state = load_project_state(root_p)
    if state is None:
        result.errors.append(
            ResultMessage(
                code="PROJECT_STATE_MISSING",
                message=(
                    f"No project state metadata found at '{state_path}'. "
                    "Run 'rag init' first."
                ),
            )
        )
        result.data = {
            "summary": {
                "project_root": str(root_p),
                "config_path": str(cfg_path),
                "state": "missing",
                "destructive": destructive,
            },
            "details": {},
        }
        return result

    # Resolve model capabilities: cache hit → no model load; miss → introspect + cache.
    # Must happen before snapshot_foundational_config so that the resolved dim is used.
    _specs_dir = root_p / ".rag" / "metadata" / "model_specs"
    _cache = ModelSpecsCache(_specs_dir)
    caps = _cache.get(cfg.embedding.model_id)
    if caps is None:
        caps = ModelInspector.introspect(
            cfg.embedding.model_id,
            cfg.embedding.model_cache_dir,
            backend=cfg.embedding.backend,
        )
        _cache.put(caps)

    # validate_for_build validates the dim, resolves it on cfg, then checks the
    # stored schema for BREAKING changes.  Dim is a concrete integer after this.
    _engine = _get_engine()
    ProjectValidator.validate_for_build(
        cfg, caps, _engine, destructive=destructive)

    current_foundation = snapshot_foundational_config(cfg)

    # First-time build after init.
    if state.state == ProjectLifecycleState.INITIALISED:
        init_database(cfg, destructive=destructive)

        new_state = create_project_state(
            cfg=cfg,
            state=ProjectLifecycleState.SET_UP,
        )
        save_project_state(new_state)

        result.data = {
            "summary": {
                "project_root": str(root_p),
                "config_path": str(cfg_path),
                "state_before": ProjectLifecycleState.INITIALISED.value,
                "state_after": ProjectLifecycleState.SET_UP.value,
                "destructive": destructive,
                "changed_foundation_fields": [],
                "database_initialized": True,
            },
            "details": {
                "project_state_path": str(state_path),
                "foundation": current_foundation.model_dump(mode="json"),
            },
        }
        result.success = True
        return result

    # Already built before: compare foundational config.
    if state.state == ProjectLifecycleState.SET_UP:
        changed_fields = changed_foundation_fields(
            current=current_foundation,
            recorded=state.foundation,
        )
        breaking_fields, stale_fields = classify_foundation_changes(
            changed_fields)

        # ── Case 1: nothing changed ──────────────────────────────────────────
        if not changed_fields:
            result.errors.append(
                ResultMessage(
                    code="ALREADY_BUILT",
                    message=(
                        "Project storage is already built and foundational config "
                        "has not changed. No action taken."
                    ),
                )
            )
            result.data = {
                "summary": {
                    "project_root": str(root_p),
                    "config_path": str(cfg_path),
                    "state_before": ProjectLifecycleState.SET_UP.value,
                    "state_after": ProjectLifecycleState.SET_UP.value,
                    "destructive": destructive,
                    "changed_foundation_fields": [],
                    "database_initialized": True,
                },
                "details": {
                    "project_state_path": str(state_path),
                    "foundation": current_foundation.model_dump(mode="json"),
                },
            }
            result.success = False
            return result

        # ── Case 2: only STALE fields changed (e.g. chunking config) ────────
        # The DB schema is unaffected — just update the snapshot so the new
        # config is recorded, and advise the user to re-ingest.
        if not breaking_fields:
            new_state = create_project_state(
                cfg=cfg,
                state=ProjectLifecycleState.SET_UP,
            )
            save_project_state(new_state)

            result.warnings.append(
                ResultMessage(
                    code="STALE_CONFIG_UPDATED",
                    message=(
                        f"Non-breaking config changed ({
                            ', '.join(stale_fields)}). "
                        "Project snapshot updated. "
                        "Run 'rag ingest --force' to re-index files with the new settings."
                    ),
                )
            )
            result.data = {
                "summary": {
                    "project_root": str(root_p),
                    "config_path": str(cfg_path),
                    "state_before": ProjectLifecycleState.SET_UP.value,
                    "state_after": ProjectLifecycleState.SET_UP.value,
                    "destructive": False,
                    "changed_foundation_fields": stale_fields,
                    "database_initialized": True,
                    "no_op": False,
                },
                "details": {
                    "project_state_path": str(state_path),
                    "recorded_foundation": state.foundation.model_dump(mode="json"),
                    "current_foundation": current_foundation.model_dump(mode="json"),
                },
            }
            result.success = True
            return result

        # ── Case 3: BREAKING fields changed — DB rebuild required ────────────
        if not destructive:
            result.errors.append(
                ResultMessage(
                    code="FOUNDATIONAL_CONFIG_CHANGED",
                    message=(
                        "Foundational configuration has changed since the last build. "
                        f"Changed fields: {', '.join(breaking_fields)}. "
                        "Re-run with --destructive to recreate storage "
                        "using the new configuration."
                    ),
                )
            )
            result.data = {
                "summary": {
                    "project_root": str(root_p),
                    "config_path": str(cfg_path),
                    "state_before": ProjectLifecycleState.SET_UP.value,
                    "state_after": ProjectLifecycleState.SET_UP.value,
                    "destructive": destructive,
                    "changed_foundation_fields": breaking_fields,
                    "database_initialized": True,
                },
                "details": {
                    "project_state_path": str(state_path),
                    "recorded_foundation": state.foundation.model_dump(mode="json"),
                    "current_foundation": current_foundation.model_dump(mode="json"),
                },
            }
            result.success = False
            return result

        result.warnings.append(
            ResultMessage(
                code="DESTRUCTIVE_REBUILD",
                message=(
                    "Foundational configuration changed. Performing destructive rebuild "
                    f"for fields: {', '.join(breaking_fields)}."
                ),
            )
        )

        init_database(cfg, destructive=True)

        new_state = create_project_state(
            cfg=cfg,
            state=ProjectLifecycleState.SET_UP,
        )
        save_project_state(new_state)

        result.data = {
            "summary": {
                "project_root": str(root_p),
                "config_path": str(cfg_path),
                "state_before": ProjectLifecycleState.SET_UP.value,
                "state_after": ProjectLifecycleState.SET_UP.value,
                "destructive": destructive,
                "changed_foundation_fields": breaking_fields,
                "database_initialized": True,
            },
            "details": {
                "project_state_path": str(state_path),
                "recorded_foundation": state.foundation.model_dump(mode="json"),
                "current_foundation": current_foundation.model_dump(mode="json"),
            },
        }
        result.success = True
        return result

    result.errors.append(
        ResultMessage(
            code="UNKNOWN_PROJECT_STATE",
            message=f"Unsupported project state '{state.state}'.",
        )
    )
    result.data = {
        "summary": {
            "project_root": str(root_p),
            "config_path": str(cfg_path),
            "state": str(state.state),
            "destructive": destructive,
        },
        "details": {
            "project_state_path": str(state_path),
        },
    }
    return result
