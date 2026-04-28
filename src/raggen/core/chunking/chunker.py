from __future__ import annotations

from .chunks import Chunk, ChunkDraft
import hashlib
import json
from raggen.core.config.project import GroupChunkingConfig, ProjectConfig
from raggen.core.parsing.parser import Document
from typing import List, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)


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
        strategy_fn: Callable[[Document, GroupChunkingConfig], list[ChunkDraft]],
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
        if group not in self.group_configs:
            available = list(self.group_configs.keys())
            raise ValueError(
                f"No chunking config registered for group '{group}'. "
                f"Available groups: {available}"
            )
        chunk_config = self.group_configs[group]
        return self._build_chunker(chunk_config)

    def _build_chunker(self, conf: GroupChunkingConfig) -> BaseChunker:
        strategy = getattr(conf, "strategy", "fixed")

        strategy_map: dict[
            str, Callable[[Document, GroupChunkingConfig], list[ChunkDraft]]
        ] = {
            "fixed": _chunk_fixed,
            "headingAware": _chunk_heading,
            "paragraphMerge": _chunk_paragraph,
            "codeAware": _chunk_code,
        }

        if strategy not in strategy_map:
            raise ChunkingError(
                f"Unknown chunking strategy: {strategy!r}. "
                f"Available strategies: {', '.join(sorted(strategy_map))}"
            )

        return StrategyChunker(conf, strategy_map[strategy])


def _validate_document(doc: Document) -> None:
    if doc is None:
        raise ChunkingError("Cannot chunk a null document.")

    text = getattr(doc, "text", None)
    if text is None:
        raise ChunkingError("Parsed document has no 'text' attribute.")

    if not isinstance(text, str):
        raise ChunkingError("Parsed document 'text' must be a string.")


def _stable_config_hash(conf: GroupChunkingConfig) -> str:
    payload = conf.to_dict()
    conf_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(conf_json.encode("utf-8")).hexdigest()


def _enrich_chunks(
    doc: Document,
    conf: GroupChunkingConfig,
    pieces: list[ChunkDraft],
) -> list[Chunk]:
    """Convert strategy output into canonical Chunk objects."""
    config_hash = _stable_config_hash(conf)
    doc_id = getattr(doc, "doc_id", "unknown")
    source = getattr(doc, "source", None)

    out: List[Chunk] = []

    for idx, draft in enumerate(pieces):

        chunk_id = f"{doc_id}:{config_hash}:{idx}"

        out.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                chunk_index=idx,
                text=draft.text,
                start_char=draft.start_char,
                end_char=draft.end_char,
                metadata=Chunk.MetaData(
                    page_start=draft.page_start,
                    heading=draft.heading,
                    section_path=draft.section_path,
                    source=source,
                ),
                stats=Chunk.Stats(
                    char_count=len(draft.text) if draft.text is not None else None,
                    token_count=None,
                ),
                config_hash=config_hash,
            )
        )

    return out


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def _chunk_fixed(doc: Document, conf: GroupChunkingConfig) -> list[ChunkDraft]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=conf.chunk_size,
        chunk_overlap=conf.overlap,
    )
    return [ChunkDraft(text=t) for t in splitter.split_text(getattr(doc, "text", ""))]


def _chunk_paragraph(doc: Document, conf: GroupChunkingConfig) -> list[ChunkDraft]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n"],
        keep_separator=False,
        chunk_size=conf.chunk_size,
        chunk_overlap=conf.overlap,
    )
    return [ChunkDraft(text=t) for t in splitter.split_text(getattr(doc, "text", ""))]


_HEADING_LEVELS = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]


def _chunk_heading(doc: Document, conf: GroupChunkingConfig) -> list[ChunkDraft]:
    """Split on Markdown headings, then sub-split oversized sections.

    Each top-level section produced by MarkdownHeaderTextSplitter becomes
    one or more ChunkDrafts depending on whether it exceeds ``chunk_size``.
    All sub-chunks inherit the section's heading path metadata so that a
    retrieval result can always report which section it came from.

    Non-markdown documents (no headings) are handled transparently: the
    splitter returns the whole text as a single section with empty metadata,
    which then goes through the same sub-splitting path as any other section.
    """
    text = getattr(doc, "text", "") or ""

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADING_LEVELS,
        strip_headers=True,
    )
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=conf.chunk_size,
        chunk_overlap=conf.overlap,
    )

    sections = md_splitter.split_text(text)
    drafts: list[ChunkDraft] = []

    for section in sections:
        meta = section.metadata  # e.g. {"H1": "Intro", "H2": "Install"}
        section_path = [
            meta[key]
            for _, key in _HEADING_LEVELS
            if key in meta
        ]
        heading = section_path[-1] if section_path else None
        section_text = section.page_content

        # Sub-split sections that exceed chunk_size
        if len(section_text) <= conf.chunk_size:
            pieces = [section_text]
        else:
            pieces = sub_splitter.split_text(section_text)

        for piece in pieces:
            drafts.append(
                ChunkDraft(
                    text=piece,
                    heading=heading,
                    section_path=section_path if section_path else None,
                )
            )

    return drafts


def _chunk_code(doc: Document, conf: GroupChunkingConfig) -> list[ChunkDraft]:
    # Not yet implemented — will be added in the codeAware step.
    # Falls back to fixed chunking.
    return _chunk_fixed(doc, conf)
