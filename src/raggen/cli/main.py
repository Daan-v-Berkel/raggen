from __future__ import annotations

import argparse
from raggen.cli.commands import init as init_cmd, ingest as ingest_cmd, query as query_cmd


def main(argv=None):
    parser = argparse.ArgumentParser(prog="raggen")
    sub = parser.add_subparsers(dest="command")

    init_p = sub.add_parser("init")
    init_p.add_argument("--non-interactive", action="store_true")
    init_p.add_argument("--force", action="store_true")
    init_p.add_argument("--root", default=".")

    ingest_p = sub.add_parser("ingest")
    ingest_p.add_argument("--config", default=".rag/config.toml")
    ingest_p.add_argument("--destructive", action="store_true")

    query_p = sub.add_parser("query", help="Query the indexed project")
    query_p.add_argument("text", type=str, help="Query text")
    query_p.add_argument("--config", type=str, default=None)
    query_p.add_argument("--top-k", type=int, default=None)

    args = parser.parse_args(argv)

    if args.command == "init":
        init_cmd.run_init(
            root=args.root, non_interactive=args.non_interactive, force=args.force)
    elif args.command == "ingest":
        ingest_cmd.run_ingest(config_path=args.config,
                              destructive=args.destructive)
    elif args.command == "query":
        return query_cmd.run_query(
            args.text,
            config_path=args.config,
            top_k=args.top_k,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
