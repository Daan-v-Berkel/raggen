from __future__ import annotations

from pathlib import Path
from datetime import datetime
from raggen.core.results.renderers import get_renderer
from raggen.core.results.projection import project_result
from raggen.core.runs.store import get_run_store
from raggen.core.bootstrap import bootstrap, BootstrapError
from raggen.core.results.formats import OutputFormat


def run_list(
    config_path: str | None = None,
    limit: int = 20,
    operation: str | None = None,
    detailed: bool = False,
    format_as: OutputFormat = OutputFormat.TEXT,
) -> int:
    try:
        bootstrap(Path(config_path) if config_path else None)
    except BootstrapError as e:
        print(f"Error: {e}")
        return 1
    store = get_run_store()
    runs = store.list_runs(limit=limit, operation=operation)
    if not detailed and len(runs) > 20:
        runs = runs[:20]

    if not runs:
        print("No runs found.")
        return 0

    print(f"{'RUN ID':<40} {'OPERATION':<12} {
          'CREATED':<32} {'OK':<6} {'WARN':<6} {'ERR':<6}")
    for run in runs:
        print(
            f"{run.run_id:<40} "
            f"{run.operation:<12} "
            f"{_parse_timestamp(run.created_at):<28} "
            f"{str(run.success):<6} "
            f"{run.n_warnings:<6} "
            f"{run.n_errors:<6}"
        )
    return 0


def run_show(
    config_path: str | None = None,
    run_id: str | None = None,
    latest: bool = False,
    operation: str | None = None,
    detailed: bool = False,
    format_as: OutputFormat = OutputFormat.JSON,
) -> int:
    try:
        bootstrap(Path(config_path) if config_path else None)
    except BootstrapError as e:
        print(f"Error: {e}")
        return 1

    store = get_run_store()

    run_id = run_id
    if latest:
        latest = store.get_latest_run(operation=operation)
        if latest is None:
            if operation:
                print(f"No runs found for operation '{operation}'.")
            else:
                print("No runs found.")
            print("Use 'rag runs list' to see available runs.")
            return 1
        run_id = latest.run_id

    if not run_id:
        print("Provide a run_id or use --latest.")
        print("Use 'rag runs list' to see available runs.")
        return 1

    result = store.load_result(run_id)
    projected = project_result(result, detailed=detailed)
    renderer = get_renderer(format_as)

    # Prepend a clear header
    header = f"─── {result.operation} ─ {
        _parse_timestamp(result.created_at)} ─ [{result.run_id}] ───"
    print(header)
    print(f"    Status: {'Success' if result.success else 'Failed'}")
    print(" output ".center(len(header), "─"))

    print(renderer.render(projected))
    return 0


def _parse_timestamp(ts: str) -> str:
    if not ts:
        return ""
    dt = datetime.fromisoformat(ts)
    dt = dt.astimezone()
    return dt.strftime("%c")
