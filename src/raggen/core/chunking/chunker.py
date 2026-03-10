from __future__ import annotations

from .chunks import Chunk
import hashlib
import json
from raggen.core.config.project import GroupChunkingConfig, ProjectConfig
from raggen.core.parsing.parser import Document
from typing import List, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkingError(RuntimeError):
    pass


class BaseChunker(ABC):
    """
    One reusable chunker instance per file group.
    """

    def __init__(self, config: GroupChunkingConfig):
        self.config = config

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        raise NotImplementedError


class StrategyChunker(BaseChunker):
    """
    Generic chunker that delegates to a specific strategy function.
    """

    def __init__(
        self,
        config: GroupChunkingConfig,
        strategy_fn: Callable[[Document, GroupChunkingConfig], list[Chunk]],
    ):
        super().__init__(config)
        self._strategy_fn = strategy_fn

    def chunk(self, doc: Document) -> list[Chunk]:
        _validate_document(doc)
        pieces = self._strategy_fn(doc, self.config)
        return _enrich_chunks(doc, self.config, pieces)


@dataclass
class ChunkerRegistry:
    """
    Registry that returns one reusable chunker per file group.

        chunker = registry.get("code")
        chunks = chunker.chunk(parsed_doc)
    """

    def __init__(self):
        cfg = ProjectConfig.get_config()
        self.group_configs = cfg.chunking
        self.fallback_group = cfg.fallback_group

    def get(self, group: str) -> BaseChunker:
        """
        Return a chunker for the given group.
        Unknown groups resolve to the configured fallback group.
        """
        chunk_config = self.group_configs[group]

        return self._build_chunker(chunk_config)

    def _build_chunker(self, conf: GroupChunkingConfig) -> BaseChunker:
        strategy = getattr(conf, "strategy", "fixed")

        strategy_map: dict[str, Callable[[Document, GroupChunkingConfig], list[Chunk]]] = {
            "fixed": _chunk_fixed,
            "headingAware": _chunk_heading,
            "paragraphMerge": _chunk_paragraph,
            "tokenAware": _chunk_token,
            "ast": _chunk_ast,
        }

        if strategy not in strategy_map:
            raise ChunkingError(f"Unknown chunking strategy: {strategy!r}")

        return StrategyChunker(conf, strategy_map[strategy])


def _validate_document(doc: Document) -> None:
    """
    Minimal sanity check for parsed documents.

    We keep this intentionally loose because the exact Document model
    may evolve independently from chunking.
    """
    if doc is None:
        raise ChunkingError("Cannot chunk a null document.")

    text = getattr(doc, "text", None)
    if text is None:
        raise ChunkingError("Parsed document has no 'text' attribute.")

    if not isinstance(text, str):
        raise ChunkingError("Parsed document 'text' must be a string.")


def _stable_config_hash(conf: GroupChunkingConfig) -> str:
    """
    Stable hash for the chunking config used to produce chunks.
    """
    payload = conf.to_dict()

    conf_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(conf_json.encode("utf-8")).hexdigest()


def _enrich_chunks(
    doc: Document,
    conf: GroupChunkingConfig,
    pieces: list[str],
) -> list[Chunk]:
    """
    Convert raw text pieces into the project's canonical Chunk objects.

    This version does not rely on char offsets.
    """
    # TODO:calculate cnfig_hash once per config
    config_hash = _stable_config_hash(conf)
    doc_id = getattr(doc, "doc_id", "unknown")
    source = getattr(doc, "source", None)

    out: List[Chunk] = []

    for idx, piece in enumerate(pieces):
        chunk_id = f"{doc_id}:{config_hash}:{idx}"

        meta = {
            "page_start": None,
            "page_end": None,
            "heading": None,
            "section_path": None,
            "source": source,
        }

        stats = {
            "char_count": len(piece) if piece is not None else None,
            "token_count": None,
        }

        out.append(
            Chunk(
                doc_id=doc_id,
                chunk_index=idx,
                text=piece,
                start_char=None,
                end_char=None,
                metadata=meta,
                stats=stats,
                config_hash=config_hash,
                chunk_id=chunk_id,
            )
        )

    return out
# ---------------------------------------------------------------------------
# Strategy hooks
# ---------------------------------------------------------------------------


def _chunk_fixed(doc: Document, conf: GroupChunkingConfig) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=conf.chunk_size,
        chunk_overlap=conf.overlap,
    )
    return splitter.split_text(getattr(doc, "text", ""))


def _chunk_heading(doc: Document, conf: GroupChunkingConfig) -> list[str]:
    # Not implemented: fallback to fixed
    return _chunk_fixed(doc, conf)


def _chunk_paragraph(doc: Document, conf: GroupChunkingConfig) -> list[str]:
    # Simple paragraph-based chunking: split on double-newline and then apply merging
    text = getattr(doc, "text", "")
    paras = text.split("\n\n") if text else []
    out: List[str] = []
    for p in paras:
        if p == "":
            continue
        if len(p) <= conf.chunk_size:
            out.append(p)
        else:
            # fallback to fixed-size slicing within paragraph
            start = 0
            while start < len(p):
                end = min(len(p), start + conf.chunk_size)
                out.append(p[start:end])
                start = max(0, end - conf.overlap)
    return out


def _chunk_ast(doc: Document, conf: GroupChunkingConfig) -> list[str]:
    # Not yet implemented
    return _chunk_fixed(doc, conf)


def _chunk_token(doc: Document, conf: GroupChunkingConfig) -> list[str]:
    # Token-aware chunking not implemented yet; fallback to fixed
    return _chunk_fixed(doc, conf)
