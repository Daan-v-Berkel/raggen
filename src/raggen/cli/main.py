from __future__ import annotations

import argparse

from raggen.cli.commands import (
    init as init_cmd,
    ingest as ingest_cmd,
    query as query_cmd,
)
from raggen.core.results.formats import OutputFormat

format_choices = [fmt.value for fmt in OutputFormat]


def build_parser() -> argparse.ArgumentParser:
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--format",
        default=OutputFormat.JSON,
        choices=format_choices,
        help=f"Output format (default: %(default)s). Options: {', '.join(format_choices)}",
    )

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
        help="Create project configuration and initialize the database.",
        parents=[common_parser],
        description=(
            "Initialize a Raggen project in the target root directory.\n"
            "This creates the .rag configuration directory and initializes the database."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    init_p.add_argument(
        "--root",
        default=".",
        help="Project root directory where .rag/ should be created.",
    )
    init_p.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run initialization without interactive prompts, using defaults.",
    )
    init_p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing initialization if one is already present.",
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

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        return init_cmd.run_init(
            root=args.root,
            non_interactive=args.non_interactive,
            force=args.force,
        )

    if args.command == "ingest":
        return ingest_cmd.run_ingest(
            config_path=args.config,
            destructive=args.destructive,
            format_as=args.format,
        )

    if args.command == "query":
        return query_cmd.run_query(
            args.text,
            config_path=args.config,
            top_k=args.top_k,
            format_as=args.format
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
