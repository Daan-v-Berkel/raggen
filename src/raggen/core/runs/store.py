from __future__ import annotations

import json
from pathlib import Path

from raggen.core.config.project import ProjectConfig
from raggen.core.results.envelope import ResultEnvelope
from raggen.core.runs.interface import RunStore, RunSummary


class FileRunStore(RunStore):
    def __init__(self, runs_root: Path):
        self.runs_root = runs_root

    def save_result(self, result: ResultEnvelope) -> Path:
        run_dir = self.runs_root / result.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        result_path = run_dir / "result.json"
        result_path.write_text(
            json.dumps(result.to_plain(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return result_path

    def load_result(self, run_id: str) -> ResultEnvelope:
        result_path = self.runs_root / run_id / "result.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return ResultEnvelope.model_validate(payload)

    def list_runs(
        self,
        *,
        limit: int | None = None,
        operation: str | None = None,
    ) -> list[RunSummary]:
        summaries: list[RunSummary] = []

        if not self.runs_root.exists():
            return []

        for run_dir in self.runs_root.iterdir():
            if not run_dir.is_dir():
                continue

            result_path = run_dir / "result.json"
            if not result_path.exists():
                continue

            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            run_operation = payload.get("operation")
            if operation is not None and run_operation != operation:
                continue

            summaries.append(
                RunSummary(
                    run_id=payload["run_id"],
                    operation=run_operation,
                    created_at=payload["created_at"],
                    success=payload["success"],
                    n_errors=len(payload.get("errors", [])),
                    n_warnings=len(payload.get("warnings", [])),
                )
            )

        summaries.sort(key=lambda r: r.created_at, reverse=True)

        if limit is not None and limit > 0:
            summaries = summaries[:limit]

        return summaries

    def get_latest_run(self, *, operation: str | None = None) -> RunSummary | None:
        runs = self.list_runs(limit=1, operation=operation)
        return runs[0] if runs else None


def get_run_store() -> RunStore:
    cfg = ProjectConfig.get_config()
    runs_root = Path(cfg.project_root) / ".rag" / "runs"
    return FileRunStore(runs_root)
