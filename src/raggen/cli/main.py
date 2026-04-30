from __future__ import annotations

import argparse
import sys

from raggen.core.results.formats import OutputFormat
from raggen.core.embeddings.config_validator import ModelCapabilityError
from raggen.core.embeddings.model_specs_cache import MissingModelSpecsError
from raggen.core.store.exceptions import SchemaMismatchError
from raggen.core.config.project import ConfigError

_USER_ERRORS = (
    ModelCapabilityError,
    MissingModelSpecsError,
    SchemaMismatchError,
    ConfigError,
)


def build_common_parser() -> argparse.ArgumentParser:
    common_parser = argparse.ArgumentParser(add_help=False)
    format_choices = [fmt.value for fmt in OutputFormat]
    common_parser.add_argument(
        "--format",
        default=OutputFormat.TEXT.value,
        choices=format_choices,
        help=f"Output format. Options: {', '.join(format_choices)}",
    )
    common_parser.add_argument(
        "--detailed",
        action="store_true",
        help=f"Output presentation, by default this is truncated, use this flag to get the full output",
    )

    return common_parser


def build_parser() -> argparse.ArgumentParser:
    common_parser = build_common_parser()

    parser = argparse.ArgumentParser(
        prog="rag",
        description=(
            "raggen — local RAG indexing and querying tool.\n"
            "Use one of the subcommands below to initialize a project, ingest files, "
            "or query the indexed content."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sub = parser.add_subparsers(
        dest="command",
        title="commands",
        metavar="<command>",
        help="Run 'rag <command> -h' for command-specific help.",
    )

    # init
    init_p = sub.add_parser(
        "init",
        parents=[common_parser],
        help="Initialise project scaffold and default configuration",
    )
    init_p.add_argument("root", type=str)
    init_p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing project scaffold. WARNING: deletes the entire .rag directory.",
    )

    # build
    build_p = sub.add_parser(
        "build",
        parents=[common_parser],
        help="Create project storage foundation from configuration",
    )
    build_p.add_argument(
        "--config",
        default=".rag/config.toml",
        help="Path to project config file.",
    )
    build_p.add_argument(
        "--destructive",
        action="store_true",
        help="Allow destructive rebuild when foundational configuration has changed.\nTHIS DELETES AND REBUILDS DATABASE",
    )

    # ingest
    ingest_p = sub.add_parser(
        "ingest",
        help="Scan files and update the index.",
        parents=[common_parser],
        description=(
            "Ingest files according to the project configuration.\n"
            "This scans the configured project, parses files, chunks them, embeds them, "
            "and updates the database."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ingest_p.add_argument(
        "--config",
        default=".rag/config.toml",
        help="Path to the project configuration TOML file.",
    )
    ingest_p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-ingest all files, even those that have not changed. "
            "Useful after chunking or embedding config changes."
        ),
    )
    ingest_p.add_argument(
        "--no-progress",
        action="store_true",
        dest="no_progress",
        help="Suppress the per-file progress line. Useful when capturing output or running non-interactively.",
    )

    # query
    query_p = sub.add_parser(
        "query",
        help="Run a similarity search against the indexed project.",
        parents=[common_parser],
        description=(
            "Query the indexed project and return the most relevant chunks.\n"
            "This performs retrieval only; generation may be added separately."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    query_p.add_argument(
        "text",
        type=str,
        help="Query text to search for in the indexed content.",
    )
    query_p.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional path to the project configuration TOML file. "
            "If omitted, the default project config is used."
        ),
    )
    query_p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Maximum number of matching chunks to return.",
    )

    # runs
    runs_p = sub.add_parser("runs", help="Inspect stored run history")
    runs_p.set_defaults(_runs_parser=runs_p)
    runs_sub = runs_p.add_subparsers(dest="runs_command")

    runs_list_p = runs_sub.add_parser(
        "list",
        parents=[common_parser],
        conflict_handler="resolve",
        help="List stored runs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    runs_list_p.add_argument(
        "--detailed",
        action="store_true",
        help="No-op for this command: list output is always the same regardless of this flag.",
    )
    runs_list_p.add_argument(
        "--config",
        default=".rag/config.toml",
        help="Path to the project configuration TOML file.",
    )
    runs_list_p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of runs to list.",
    )
    runs_list_p.add_argument(
        "--operation",
        type=str,
        default=None,
        help="Filter by operation, e.g. ingest or query.",
    )

    runs_show_p = runs_sub.add_parser(
        "show",
        parents=[common_parser],
        help="Show a stored run result",
    )
    runs_show_p.add_argument(
        "--config",
        default=".rag/config.toml",
        help="Path to the project configuration TOML file.",
    )
    runs_show_p.add_argument(
        "run_id",
        nargs="?",
        default=None,
        help="Run ID to show.",
    )
    runs_show_p.add_argument(
        "--latest",
        action="store_true",
        help="Show the latest run, optionally filtered by --operation.",
    )
    runs_show_p.add_argument(
        "--operation",
        type=str,
        default=None,
        help="Filter latest lookup by operation, e.g. ingest or query.",
    )

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return _dispatch(args, parser)
    except _USER_ERRORS as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _dispatch(args, parser):
    if args.command == "init":
        from raggen.cli.commands import init as init_cmd
        return init_cmd.run_init(
            root=args.root,
            force=args.force,
            detailed=args.detailed,
            format_as=args.format,
        )

    if args.command == "build":
        from raggen.cli.commands import build as build_cmd
        return build_cmd.run_build(
            config=args.config,
            destructive=args.destructive,
            detailed=args.detailed,
            format_as=args.format,
        )

    if args.command == "ingest":
        from raggen.cli.commands import ingest as ingest_cmd
        return ingest_cmd.run_ingest(
            config_path=args.config,
            force=args.force,
            detailed=args.detailed,
            format_as=args.format,
            no_progress=args.no_progress,
        )

    if args.command == "query":
        from raggen.cli.commands import query as query_cmd
        return query_cmd.run_query(
            args.text,
            config_path=args.config,
            top_k=args.top_k,
            detailed=args.detailed,
            format_as=args.format,
        )

    if args.command == "runs":
        from raggen.cli.commands import runs as runs_cmd
        if args.runs_command == "list":
            return runs_cmd.run_list(
                config_path=args.config,
                limit=args.limit,
                operation=args.operation,
                detailed=args.detailed,
                format_as=args.format,
            )
        if args.runs_command == "show":
            return runs_cmd.run_show(
                config_path=args.config,
                run_id=args.run_id,
                latest=args.latest,
                operation=args.operation,
                detailed=args.detailed,
                format_as=args.format,
            )
        # `rag runs` with no subcommand — show runs-specific help
        args._runs_parser.print_help()
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
