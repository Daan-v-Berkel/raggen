#!/usr/bin/env python3
"""
Lightweight end-to-end script: scan project tree, parse files with the fallback parser,
then chunk each parsed document using DEFAULT_CHUNK_CONFIG.

Usage:
    python scripts/e2e_chunk_test.py /path/to/project/root [ignorefile]

Arguments:
  root      - project root to scan (first arg)
  ignorefile- optional path to an ignore file (gitignore-style globs, one per line)

Behavior:
  1) Scans files under `root`, respecting built-in skips and optional ignorefile patterns.
  2) Parses each file using the ParserRegistry (PlainTextFallbackParser as fallback).
  3) Chunks each parsed Document with DEFAULT_CHUNK_CONFIG and prints a summary.

This script is intended for local manual testing only.
"""
from raggen.core.parsing.parser import ParserRegistry, ParseInput, ParserService
from raggen.core.parsing.PlainTextParser import PlainTextFallbackParser
from raggen.core.chunking.chunks import DEFAULT_CHUNK_CONFIG
from raggen.core.chunking.chunker import Chunker
import argparse
import os
import sys
from pathlib import Path
import fnmatch

# Ensure repo root is importable when running this script directly
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


BUILTIN_SKIP_DIRS = {".git", ".venv", "venv", "env",
                     "__pycache__", "logs", ".rag", "node_modules"}


def load_ignore_patterns(ignorefile: Path):
    if not ignorefile or not ignorefile.exists():
        return []
    patterns = []
    for line in ignorefile.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        patterns.append(s)
    return patterns


def should_ignore(path: Path, root: Path, patterns: list[str]) -> bool:
    """Return True if path should be ignored (patterns are gitignore-style globs).

    Matching is done against the POSIX relative path from root, and also against the filename.
    """
    try:
        rel = str(path.relative_to(root)).replace(os.sep, '/')
    except Exception:
        rel = str(path)

    for pat in patterns:
        # If pattern ends with '/', treat as directory prefix
        if pat.endswith('/') and rel.startswith(pat.rstrip('/')):
            return True
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(path.name, pat):
            return True
    return False


def scan_files(root: Path, ignore_patterns: list[str]):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # skip built-in directories
        dirnames[:] = [
            d for d in dirnames if d not in BUILTIN_SKIP_DIRS and not d.startswith('.')]
        for fn in filenames:
            if fn.startswith('.'):
                continue
            p = Path(dirpath) / fn
            # skip the script itself
            if p.resolve() == Path(__file__).resolve():
                continue
            if should_ignore(p, root, ignore_patterns):
                continue
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', nargs='?', default=str(Path.cwd()),
                    help='Project root to scan (first arg)')
    ap.add_argument('ignorefile', nargs='?',
                    help='Optional ignore file (second arg)')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        print('Provided root is not a directory')
        sys.exit(1)

    ignorefile = Path(args.ignorefile).resolve() if args.ignorefile else None
    patterns = load_ignore_patterns(ignorefile) if ignorefile else []
    print(f"Scanning {root} ... (ignore patterns: {len(patterns)})")

    files = scan_files(root, patterns)
    print(f"Found {len(files)} files")

    # Prepare parser registry and service
    registry = ParserRegistry(fallback_parser=PlainTextFallbackParser())
    parser_service = ParserService(registry)

    parsed = {}
    for p in files:
        try:
            data = p.read_bytes()
        except Exception as e:
            print(f"Skipping {p}: could not read ({e})")
            continue
        doc_id = str(p.relative_to(root))
        mimetype = 'application/octet-stream'
        filename = doc_id

        try:
            inp = ParseInput(doc_id=doc_id, data=data,
                             mimetype=mimetype, filename=filename)
        except Exception as e:
            print(f"Failed to build ParseInput for {doc_id}: {e}")
            continue

        try:
            result = parser_service.parse_document(inp)
        except Exception as e:
            print(f"Parsing failed for {doc_id}: {e}")
            continue

        parsed[doc_id] = result

    print(f"Parsed {len(parsed)} documents")

    total_chunks = 0
    for doc_id, result in parsed.items():
        doc = result.document
        # ensure source populated
        if getattr(doc, 'source', None) is None:
            try:
                doc.source = result.document.doc_id
            except Exception:
                pass
        chunker = Chunker(doc)
        try:
            chunks = chunker.chunk(DEFAULT_CHUNK_CONFIG)
        except Exception as e:
            print(f"Chunking failed for {doc_id}: {e}")
            continue
        count = len(chunks)
        total_chunks += count
        print(f"Document: {doc_id} -> {count} chunks")

    print(f"Produced {total_chunks} chunks from {len(parsed)} documents")
    print(f"Example Chunk: {chunks[0]}")


if __name__ == '__main__':
    main()
