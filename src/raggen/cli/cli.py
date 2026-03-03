import argparse
from pathlib import Path
from .workspace import init_workspace


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    init_cmd = sub.add_parser("init")
    init_cmd.add_argument("--force", action="store_true")

    args = parser.parse_args()

    if args.command == "init":
        init_workspace(Path.cwd(), force=args.force)
        print("Initialized .rag workspace")


if __name__ == "__main__":
    main()
