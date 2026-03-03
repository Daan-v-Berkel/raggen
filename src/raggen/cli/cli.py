import argparse
from pathlib import Path
from .workspace import init_workspace, DEFAULT_CONFIG


def _prompt(prompt: str, default: str | None = None) -> str:
    if default is not None:
        full = f"{prompt} [{default}]: "
    else:
        full = f"{prompt}: "
    v = input(full).strip()
    return v or (default or "")


def _choose(prompt: str, choices: list[str], default: str) -> str:
    choices_str = ", ".join(choices)
    while True:
        v = _prompt(f"{prompt} ({choices_str})", default)
        if v in choices:
            return v
        print(f"Invalid choice: {v}. Choose one of: {choices_str}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    init_cmd = sub.add_parser("init")
    init_cmd.add_argument("--force", action="store_true")

    args = parser.parse_args()

    if args.command == "init":
        # Interactive installer
        root_raw = _prompt("Root directory to initialize", ".")
        root = Path(root_raw).expanduser().resolve()

        ignore_file = _prompt("Ignore file to use", ".gitignore")

        content_types = ["code", "documentation", "research", "contracts", "mixed"]
        content_default = "mixed"
        content_type = _choose("Type of content to embed", content_types, content_default)

        backends = ["sqlite_vec", "sqlite", "postgres"]
        backend_default = DEFAULT_CONFIG.get("vector_backend", "sqlite_vec")
        backend = _choose("Vector/backend to use", backends, backend_default)

        # Sensible defaults per content type
        model_map = {
            "code": "sentence-transformers/all-MiniLM-L6-v2",
            "documentation": "bge-small-en",
            "research": "bge-small-en",
            "contracts": "bge-small-en",
            "mixed": "bge-small-en",
        }

        chunk_map = {
            "code": {"chunk_size": 800, "overlap": 100},
            "documentation": {"chunk_size": 1000, "overlap": 150},
            "research": {"chunk_size": 1500, "overlap": 200},
            "contracts": {"chunk_size": 1000, "overlap": 150},
            "mixed": {"chunk_size": 1000, "overlap": 150},
        }

        chosen_model = model_map.get(content_type, DEFAULT_CONFIG.get("embedding_model"))
        chosen_chunking = chunk_map.get(content_type, DEFAULT_CONFIG.get("chunking", {}))

        config = dict(DEFAULT_CONFIG)
        config["vector_backend"] = backend
        config["embedding_model"] = chosen_model
        # Ensure chunking exists and merge
        conf_chunk = dict(config.get("chunking", {}))
        conf_chunk.update(chosen_chunking)
        config["chunking"] = conf_chunk
        config["ignore_file"] = ignore_file
        config["content_type"] = content_type

        print("\nSummary:\n")
        print(f"Root: {root}")
        print(f"Ignore file: {ignore_file}")
        print(f"Content type: {content_type}")
        print(f"Embedding model: {chosen_model}")
        print(f"Chunking: {config['chunking']}")
        print(f"Backend: {backend}\n")

        confirm = _prompt("Proceed and initialize workspace? (y/n)", "y").lower()
        if confirm not in ("y", "yes"):
            print("Aborted.")
            return

        init_workspace(root, force=args.force, config=config)
        print(f"Initialized .rag workspace at {root / '.rag'}")


if __name__ == "__main__":
    main()
