from __future__ import annotations

from pathlib import Path
from typing import Protocol

from raggen.core.results.envelope import ResultEnvelope


class RunStore(Protocol):
    def save_result(self, result: ResultEnvelope) -> Path:
        """Persist a result envelope and return the written file path."""

    def load_result(self, run_id: str) -> ResultEnvelope:
        """Load a previously persisted result envelope by run_id."""
