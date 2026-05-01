from __future__ import annotations

from raggen.core.metadata.models import FoundationalConfigSnapshot

# Fields in FoundationalConfigSnapshot whose change requires a full DB rebuild.
# Everything else (currently just chunking_hashes) is STALE — the data can be
# brought back in sync by re-ingesting without touching the schema.
_BREAKING_SNAPSHOT_FIELDS: frozenset[str] = frozenset({
    "schema_version",
    "embedding_model",
    "embedding_backend",
    "embedding_dim",
    "storage_backend_key",
    "database_url",
    "vector_backend_import",
})


def changed_foundation_fields(
    current: FoundationalConfigSnapshot,
    recorded: FoundationalConfigSnapshot,
) -> list[str]:
    current_map = current.model_dump()
    recorded_map = recorded.model_dump()

    changed: list[str] = []
    for key, value in current_map.items():
        if recorded_map.get(key) != value:
            changed.append(key)
    return changed


def classify_foundation_changes(
    changed: list[str],
) -> tuple[list[str], list[str]]:
    """Split a list of changed snapshot field names into (breaking, stale).

    Breaking fields require a destructive DB rebuild.
    Stale fields (currently only chunking_hashes) can be resolved by
    re-ingesting without touching the schema.
    """
    breaking = [f for f in changed if f in _BREAKING_SNAPSHOT_FIELDS]
    stale = [f for f in changed if f not in _BREAKING_SNAPSHOT_FIELDS]
    return breaking, stale


def foundation_changed(
    current: FoundationalConfigSnapshot,
    recorded: FoundationalConfigSnapshot,
) -> bool:
    return bool(changed_foundation_fields(current, recorded))
