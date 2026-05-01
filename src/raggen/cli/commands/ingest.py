from __future__ import annotations

import sys
import threading
from pathlib import Path
from raggen.core.bootstrap import bootstrap, BootstrapError
from raggen.core.results.formats import OutputFormat
from raggen.core.results.renderers import get_renderer
from raggen.core.results.projection import project_result


def run_ingest(
    *, config_path: str = ".rag/config.toml",
    force: bool = False,
    format_as: OutputFormat = OutputFormat.JSON,
    detailed: bool = False,
    no_progress: bool = False,
) -> int:
    try:
        bootstrap(Path(config_path) if config_path else None)
    except BootstrapError as e:
        print(f"Error: {e}")
        return 1

    # Progress display — stderr only, suppressed when not a TTY or --no-progress.
    # Agents / CI typically don't have a TTY so they get silence automatically.
    show_progress = not no_progress and sys.stderr.isatty()

    if show_progress:
        from raggen.core.ingest.ingest_service import do_ingest
        from raggen.core.config.project import ProjectConfig
        from raggen.core.scanner import scan_files

        cfg = ProjectConfig.get_config()
        scanned = scan_files(
            cfg.project_root,
            ignore_filenames=cfg.scan.ignore_files,
            ignore_patterns=cfg.scan.ignore,
        )
        total = sum(len(refs) for refs in scanned.groups.values())
        counter = [0]
        done = threading.Event()

        def on_file() -> None:
            counter[0] += 1

        result_holder: list = [None]
        exc_holder: list = [None]

        def _ingest_thread() -> None:
            try:
                result_holder[0] = do_ingest(force=force, on_file=on_file)
            except Exception as e:
                exc_holder[0] = e
            finally:
                done.set()

        t = threading.Thread(target=_ingest_thread, daemon=True)
        t.start()

        # Redraw the progress line every 100 ms until ingest finishes.
        while not done.wait(timeout=0.1):
            sys.stderr.write(f"\rIngesting... [{counter[0]}/{total}]")
            sys.stderr.flush()

        sys.stderr.write(f"\rIngesting... [{counter[0]}/{total}]\n")
        sys.stderr.flush()
        t.join()

        if exc_holder[0] is not None:
            raise exc_holder[0]

        result = result_holder[0]

    else:
        from raggen.core.ingest.ingest_service import do_ingest
        result = do_ingest(force=force)

    projected = project_result(result, detailed=detailed)
    renderer = get_renderer(format_as)
    print(renderer.render(projected))
    return 0 if result.success else 1
