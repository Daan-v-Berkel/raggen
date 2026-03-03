#!/usr/bin/env python3
"""
Lightweight end-to-end script: scan project tree, parse files with the fallback parser,
then chunk each parsed document using DEFAULT_CHUNK_CONFIG, and (NEW) embed the chunks
with a local CPU sentence-transformers model.

Usage:
    python scripts/e2e_chunk_test.py /path/to/project/root [ignorefile]

Arguments:
  root      - project root to scan (first arg)
  ignorefile- optional path to an ignore file (gitignore-style globs, one per line)

Behavior:
  1) Scans files under `root`, respecting built-in skips and optional ignorefile patterns.
  2) Parses each file using the ParserRegistry (PlainTextFallbackParser as fallback).
  3) Chunks each parsed Document with DEFAULT_CHUNK_CONFIG and prints a summary.
  4) (NEW) Embeds all chunks locally and prints embedding summary.

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
from dataclasses import dataclass
from typing import Any, Sequence, Optional, List, Dict, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "Missing dependency: sentence-transformers. Install with:\n"
        "  pip install sentence-transformers"
    ) from e


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


# ----------------------------
# NEW: minimal local embeddings
# ----------------------------

@dataclass(frozen=True)
class EmbeddingResult:
    chunk_id: str
    vector: np.ndarray  # shape: (dim,)


class LocalEmbedder:
    """
    Simple CPU embedder via sentence-transformers.
    Default model: sentence-transformers/all-MiniLM-L6-v2
    """

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_id = model_id
        self._model = SentenceTransformer(model_id, device="cpu")

    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        vecs = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
        )
        return np.asarray(vecs, dtype=np.float32)


class NpyDirCache:
    """
    Dirt-simple per-vector .npy cache under a directory.
    Keyed by sha256(model_id::chunk_id).
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, chunk_id: str, model_id: str) -> str:
        import hashlib
        raw = f"{model_id}::{chunk_id}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(self, chunk_id: str, model_id: str) -> Optional[np.ndarray]:
        p = self.root / f"{self._key(chunk_id, model_id)}.npy"
        if not p.exists():
            return None
        return np.load(p)

    def put(self, chunk_id: str, model_id: str, vector: np.ndarray) -> None:
        p = self.root / f"{self._key(chunk_id, model_id)}.npy"
        np.save(p, np.asarray(vector, dtype=np.float32))


def _chunk_text(ch: Any) -> str:
    """
    Tries common fields for your chunk objects.
    Adjust here if your chunk model uses different names.
    """
    for attr in ("text", "content", "chunk", "page_content"):
        if hasattr(ch, attr):
            v = getattr(ch, attr)
            if isinstance(v, str):
                return v
    # fallback
    return str(ch)


def _chunk_id(doc_id: str, idx: int, ch: Any) -> str:
    """
    Prefer an existing chunk_id; otherwise synthesize one deterministically.
    """
    if hasattr(ch, "chunk_id") and isinstance(getattr(ch, "chunk_id"), str):
        return getattr(ch, "chunk_id")
    return f"{doc_id}::chunk::{idx}"


def embed_chunks(
    embedder: LocalEmbedder,
    chunks: Sequence[Any],
    *,
    doc_ids_for_chunks: Sequence[str],
    batch_size: int = 32,
    normalize: bool = True,
    cache: Optional[NpyDirCache] = None,
) -> List[EmbeddingResult]:
    """
    Embeds chunk texts and returns [(chunk_id, vector)] in the same order as `chunks`.

    `doc_ids_for_chunks` must align 1:1 with `chunks` (same length).
    """
    if len(chunks) != len(doc_ids_for_chunks):
        raise ValueError(
            "doc_ids_for_chunks must align with chunks (same length).")

    results: List[EmbeddingResult] = []
    missing_ids: List[str] = []
    missing_texts: List[str] = []
    missing_positions: List[int] = []

    # Pre-allocate result slots to preserve order
    slots: List[Optional[EmbeddingResult]] = [None] * len(chunks)

    for i, (doc_id, ch) in enumerate(zip(doc_ids_for_chunks, chunks)):
        cid = _chunk_id(doc_id, i, ch)
        txt = _chunk_text(ch).strip()

        if not txt:
            # skip empty chunks but keep placeholder vector-less result? (here: we skip)
            # If you prefer strictness, raise instead.
            continue

        if cache is not None:
            cached = cache.get(cid, embedder.model_id)
            if cached is not None:
                vec = np.asarray(cached, dtype=np.float32).reshape(-1)
                slots[i] = EmbeddingResult(chunk_id=cid, vector=vec)
                continue

        missing_ids.append(cid)
        missing_texts.append(txt)
        missing_positions.append(i)

    if missing_texts:
        mat = embedder.embed_texts(
            missing_texts, batch_size=batch_size, normalize=normalize)
        if mat.shape[0] != len(missing_ids):
            raise RuntimeError("Embedding output row count mismatch.")

        for cid, pos, vec in zip(missing_ids, missing_positions, mat):
            if cache is not None:
                cache.put(cid, embedder.model_id, vec)
            slots[pos] = EmbeddingResult(
                chunk_id=cid, vector=np.asarray(vec, dtype=np.float32).reshape(-1))

    # Compact while keeping original order (skipped empties will be None)
    for item in slots:
        if item is not None:
            results.append(item)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', nargs='?', default=str(Path.cwd()),
                    help='Project root to scan (first arg)')
    ap.add_argument('ignorefile', nargs='?',
                    help='Optional ignore file (second arg)')

    # NEW embedding args
    ap.add_argument('--embed', action='store_true',
                    help='Embed chunks after chunking')
    ap.add_argument('--embed-model', default="sentence-transformers/all-MiniLM-L6-v2",
                    help='Sentence-transformers model id')
    ap.add_argument('--embed-batch', type=int, default=32,
                    help='Embedding batch size')
    ap.add_argument('--no-normalize', action='store_true',
                    help='Disable embedding normalization')
    ap.add_argument('--embed-cache-dir', default=".rag_cache/embeddings",
                    help='Directory for simple .npy embedding cache')

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
    last_chunks = None

    # NEW: collect all chunks so we can embed in one go (better batching)
    all_chunks: List[Any] = []
    all_chunk_doc_ids: List[str] = []

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

        last_chunks = chunks
        all_chunks.extend(chunks)
        all_chunk_doc_ids.extend([doc_id] * len(chunks))

    print(f"Produced {total_chunks} chunks from {len(parsed)} documents")

    if last_chunks and len(last_chunks) > 0:
        print(f"Example Chunk: {last_chunks[0]}")
    else:
        print("Example Chunk: (none)")

    # -----------------
    # NEW: embed step
    # -----------------
    if args.embed:
        if not all_chunks:
            print("No chunks to embed.")
            return

        embedder = LocalEmbedder(model_id=args.embed_model)
        cache = NpyDirCache(
            args.embed_cache_dir) if args.embed_cache_dir else None

        print("\nEmbedding chunks...")
        emb = embed_chunks(
            embedder,
            all_chunks,
            doc_ids_for_chunks=all_chunk_doc_ids,
            batch_size=args.embed_batch,
            normalize=(not args.no_normalize),
            cache=cache,
        )

        # basic summary
        dim = embedder.dim()
        print(f"Embedded {len(emb)} chunks")
        print(f"Model: {embedder.model_id}")
        print(f"Dim: {dim}")
        print(
            f"First vector shape/dtype: {emb[0].vector.shape} / {emb[0].vector.dtype}")
        print(f"First chunk_id: {emb[0].chunk_id}")


if __name__ == '__main__':
    main()
