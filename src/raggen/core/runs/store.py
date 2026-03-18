from __future__ import annotations

import json
from pathlib import Path

from raggen.core.config.project import ProjectConfig
from raggen.core.results.envelope import ResultEnvelope
from raggen.core.runs.interface import RunStore


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


def get_run_store() -> RunStore:
    cfg = ProjectConfig.get_config()
    runs_root = Path(cfg.project_root) / ".rag" / "runs"
    return FileRunStore(runs_root)
