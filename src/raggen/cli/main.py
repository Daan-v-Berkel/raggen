from __future__ import annotations

import argparse
from .commands import init as init_cmd, ingest as ingest_cmd


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(dest="command")

    init_p = sub.add_parser("init")
    init_p.add_argument("--non-interactive", action="store_true")
    init_p.add_argument("--force", action="store_true")
    init_p.add_argument("--root", default=".")

    ingest_p = sub.add_parser("ingest")
    ingest_p.add_argument("--config", default=".rag/config.toml")

    args = parser.parse_args(argv)

    if args.command == "init":
        init_cmd.run_init(root=args.root, non_interactive=args.non_interactive, force=args.force)
    elif args.command == "ingest":
        ingest_cmd.run_ingest(config_path=args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
