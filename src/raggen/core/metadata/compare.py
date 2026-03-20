from __future__ import annotations

from raggen.core.metadata.models import FoundationalConfigSnapshot


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


def foundation_changed(
    current: FoundationalConfigSnapshot,
    recorded: FoundationalConfigSnapshot,
) -> bool:
    return bool(changed_foundation_fields(current, recorded))
