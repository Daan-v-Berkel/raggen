from __future__ import annotations

import textwrap
from typing import NamedTuple

from raggen.core.config.drift_tiers import DriftTier


class FieldChange(NamedTuple):
    """One changed field, carrying everything needed to format the error."""
    field: str       # config-path form, e.g. "embedding.model_id"
    old_value: object  # value stored in the project (what was built)
    new_value: object  # value from current config
    tier: DriftTier
    reason: str      # human-readable explanation from drift_tiers


def _fmt_value(v: object) -> str:
    """Render a value for display: strings get quotes, everything else is str()."""
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)


class SchemaMismatchError(Exception):
    """
    Raised when foundational config has changed in a way that is incompatible
    with the existing schema (BREAKING tier).

    ``str(error)`` produces a human-readable block suitable for direct CLI output
    with no stack trace.  The block is computed once at construction and is
    idempotent on subsequent calls.
    """

    def __init__(self, changes: list[FieldChange]) -> None:
        self.changes = list(changes)
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        if not self.changes:
            return "Configuration change requires a full rebuild."

        lines: list[str] = [
            "Configuration change requires a full rebuild.",
            "",
            "  What changed:",
        ]

        # Align the arrow column to the longest field name.
        col_w = max(len(c.field) for c in self.changes)
        for ch in self.changes:
            old = _fmt_value(ch.old_value)
            new = _fmt_value(ch.new_value)
            lines.append(f"    {ch.field:<{col_w}}  {old} → {new}")

        lines.append("")
        lines.append("  Why this matters:")

        # Use the reason for the first (most prominent) change.
        # All changes passed here are BREAKING, so any choice is equally severe.
        reason = self.changes[0].reason
        for sentence in textwrap.wrap(reason, width=72):
            lines.append(f"    {sentence}")

        lines.extend([
            "",
            "  To proceed:",
            "    rag build --destructive",
            "",
            "  Warning: --destructive deletes all indexed data. "
            "You will need to re-ingest.",
        ])

        return "\n".join(lines)


class AlreadyInitializedError(Exception):
    pass


class InvalidConfigError(Exception):
    pass


class BackendLoadError(Exception):
    pass


class BackendNotSupportedError(Exception):
    pass


class VectorSchemaError(Exception):
    pass
