from __future__ import annotations

from pathlib import Path
from typing import Protocol

from raggen.core.results.envelope import ResultEnvelope
from pydantic import BaseModel


class RunSummary(BaseModel):
    run_id: str
    operation: str
    created_at: str
    success: bool
    n_errors: int
    n_warnings: int


class RunStore(Protocol):
    def save_result(self, result: ResultEnvelope) -> Path:
        """Persist a result envelope and return the written file path."""

    def load_result(self, run_id: str) -> ResultEnvelope:
        """Load a previously persisted result envelope by run_id."""

    def list_runs(
        self,
        *,
        limit: int | None = None,
        operation: str | None = None,
    ) -> list[RunSummary]:
        """get and return a list of RunSummary from stored runs"""

    def get_latest_run(self, *, operation: str | None = None) -> RunSummary | None:
        """get and return the latest RunSummary of type 'operation' from stored runs"""
