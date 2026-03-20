from __future__ import annotations

import argparse

from raggen.cli.commands import (
    init as init_cmd,
    ingest as ingest_cmd,
    query as query_cmd,
    runs as runs_cmd
)
from raggen.core.results.formats import OutputFormat


def build_common_parser() -> argparse.ArgumentParser:
    common_parser = argparse.ArgumentParser(add_help=False)
    format_choices = [fmt.value for fmt in OutputFormat]
    common_parser.add_argument(
        "--format",
        default=OutputFormat.JSON,
        choices=format_choices,
        help=f"Output format (default: json). Options: {', '.join(format_choices)}",
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
        prog="raggen",
        description=(
            "Raggen is a local RAG indexing and querying tool.\n"
            "Use one of the subcommands below to initialize a project, ingest files, "
            "or query the indexed content."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sub = parser.add_subparsers(
        dest="command",
        title="commands",
        metavar="<command>",
        help="Run 'raggen <command> -h' for command-specific help.",
    )

    # init
    init_p = sub.add_parser(
        "init",
        parents=[common_parser],
        help="Initialise project scaffold and default configuration",
    )
    init_p.add_argument("root", type=str)
    init_p.add_argument("--force", action="store_true")

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
        "--destructive",
        action="store_true",
        help=(
            "Rebuild database state destructively before ingesting. "
            "Use with care, as this WILL remove existing indexed data."
        ),
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
    runs_sub = runs_p.add_subparsers(dest="runs_command")

    runs_list_p = runs_sub.add_parser(
        "list",
        help="List stored runs",
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
    runs_list_p.set_defaults(func=runs_cmd.run_list)

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
        help="Show the latest run, optionally filtered by --action.",
    )
    runs_show_p.add_argument(
        "--operation",
        type=str,
        default=None,
        help="Filter latest lookup by operation, e.g. ingest or query.",
    )
    runs_show_p.set_defaults(func=runs_cmd.run_show)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        return init_cmd.run_init(
            root=args.root,
            force=args.force,
            detailed=args.detailed,
            format_as=args.format
        )

    if args.command == "ingest":
        return ingest_cmd.run_ingest(
            config_path=args.config,
            destructive=args.destructive,
            detailed=args.detailed,
            format_as=args.format
        )

    if args.command == "query":
        return query_cmd.run_query(
            args.text,
            config_path=args.config,
            top_k=args.top_k,
            detailed=args.detailed,
            format_as=args.format
        )

    if args.command == "runs":
        if args.runs_command == "list":
            return runs_cmd.run_list(config_path=args.config,
                                     limit=args.limit, operation=args.operation,
                                     detailed=args.detailed,
                                     format_as=args.format
                                     )

        elif args.runs_command == "show":
            return runs_cmd.run_show(config_path=args.config, run_id=args.run_id, latest=args.latest,
                                     operation=args.operation, detailed=args.detailed, format_as=args.format)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
