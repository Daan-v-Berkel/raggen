from __future__ import annotations

from .chunks import Chunk, ChunkDraft
import hashlib
import json
from pathlib import Path
from raggen.core.config.project import GroupChunkingConfig, ProjectConfig
from raggen.core.parsing.parser import Document
from typing import List, Callable, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

# Type aliases for readability
_LengthFn = Callable[[str], int]
_StrategyFn = Callable[[Document, GroupChunkingConfig, _LengthFn], list[ChunkDraft]]


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

    ``length_function`` controls how chunk sizes are measured:
    - ``len`` (default) — character count, matches ``unit = "chars"``
    - a tokenizer-based callable — token count, matches ``unit = "tokens"``
    """

    def __init__(
        self,
        config: GroupChunkingConfig,
        strategy_fn: _StrategyFn,
        length_function: _LengthFn = len,
    ):
        super().__init__(config)
        self._strategy_fn = strategy_fn
        self._length_function = length_function

    def chunk(self, doc: Document) -> list[Chunk]:
        _validate_document(doc)
        pieces = self._strategy_fn(doc, self.config, self._length_function)
        return _enrich_chunks(doc, self.config, pieces)


@dataclass
class ChunkerRegistry:
    """
    Registry that returns one reusable chunker per file group.

        chunker = registry.get("code")
        chunks = chunker.chunk(parsed_doc)

    Pass ``length_function`` when ``unit = "tokens"`` is configured for a
    group — typically obtained from the embedder via
    ``embedder.get_length_function()``.
    """

    def __init__(self):
        cfg = ProjectConfig.get_config()
        self.group_configs = cfg.chunking
        self.fallback_group = cfg.fallback_group

    def get(self, group: str, length_function: _LengthFn = len) -> BaseChunker:
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
        return self._build_chunker(chunk_config, length_function)

    def _build_chunker(
        self, conf: GroupChunkingConfig, length_function: _LengthFn = len
    ) -> BaseChunker:
        strategy = getattr(conf, "strategy", "fixed")

        strategy_map: dict[str, _StrategyFn] = {
            "fixed": _chunk_fixed,
            "headingAware": _chunk_heading,
            "paragraphMerge": _chunk_paragraph,
            "codeAware": _chunk_code,
        }

        return StrategyChunker(conf, strategy_map[strategy], length_function)


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


def _chunk_fixed(
    doc: Document, conf: GroupChunkingConfig, length_fn: _LengthFn = len
) -> list[ChunkDraft]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=conf.chunk_size,
        chunk_overlap=conf.overlap,
        length_function=length_fn,
    )
    return [ChunkDraft(text=t) for t in splitter.split_text(getattr(doc, "text", ""))]


def _chunk_paragraph(
    doc: Document, conf: GroupChunkingConfig, length_fn: _LengthFn = len
) -> list[ChunkDraft]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n"],
        keep_separator=False,
        chunk_size=conf.chunk_size,
        chunk_overlap=conf.overlap,
        length_function=length_fn,
    )
    return [ChunkDraft(text=t) for t in splitter.split_text(getattr(doc, "text", ""))]


_HEADING_LEVELS = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]


def _chunk_heading(
    doc: Document, conf: GroupChunkingConfig, length_fn: _LengthFn = len
) -> list[ChunkDraft]:
    """Split on Markdown headings, then sub-split oversized sections.

    Each top-level section produced by MarkdownHeaderTextSplitter becomes
    one or more ChunkDrafts depending on whether it exceeds ``chunk_size``.
    All sub-chunks inherit the section's heading path metadata so that a
    retrieval result can always report which section it came from.

    Non-markdown documents (no headings) are handled transparently: the
    splitter returns the whole text as a single section with empty metadata,
    which then goes through the same sub-splitting path as any other section.

    ``length_fn`` is used for both the size check and the sub-splitter so that
    ``unit = "tokens"`` is respected end-to-end.
    """
    text = getattr(doc, "text", "") or ""

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADING_LEVELS,
        strip_headers=True,
    )
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=conf.chunk_size,
        chunk_overlap=conf.overlap,
        length_function=length_fn,
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

        # Sub-split sections that exceed chunk_size (measured by length_fn)
        if length_fn(section_text) <= conf.chunk_size:
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


# ---------------------------------------------------------------------------
# codeAware helpers
# ---------------------------------------------------------------------------

# Maps file extensions to LangChain Language enum values so the splitter can
# use language-specific separators (e.g. \nclass , \ndef  for Python).
_EXT_TO_LANGUAGE: dict[str, Language] = {
    # Python
    ".py": Language.PYTHON,
    ".pyw": Language.PYTHON,
    # JavaScript / TypeScript
    ".js": Language.JS,
    ".mjs": Language.JS,
    ".cjs": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    # Go
    ".go": Language.GO,
    # Rust
    ".rs": Language.RUST,
    # Ruby
    ".rb": Language.RUBY,
    # Java
    ".java": Language.JAVA,
    # C / C++
    ".c": Language.C,
    ".h": Language.C,
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".cxx": Language.CPP,
    ".hpp": Language.CPP,
    # C#
    ".cs": Language.CSHARP,
    # Swift
    ".swift": Language.SWIFT,
    # Kotlin
    ".kt": Language.KOTLIN,
    ".kts": Language.KOTLIN,
    # Scala
    ".scala": Language.SCALA,
    # PHP
    ".php": Language.PHP,
    # R
    ".r": Language.R,
    # Lua
    ".lua": Language.LUA,
    # Perl
    ".pl": Language.PERL,
    ".pm": Language.PERL,
    # Elixir
    ".ex": Language.ELIXIR,
    ".exs": Language.ELIXIR,
    # Haskell
    ".hs": Language.HASKELL,
    # Solidity
    ".sol": Language.SOL,
    # Protobuf
    ".proto": Language.PROTO,
    # PowerShell
    ".ps1": Language.POWERSHELL,
}


def _detect_language(doc: Document) -> Optional[Language]:
    """Return the Language enum for the document's file extension, or None."""
    source = getattr(doc, "source", None)
    rel_path = getattr(source, "rel_path", None) if source else None
    if not rel_path:
        return None
    return _EXT_TO_LANGUAGE.get(Path(rel_path).suffix.lower())


def _chunk_code(
    doc: Document, conf: GroupChunkingConfig, length_fn: _LengthFn = len
) -> list[ChunkDraft]:
    """Split code files using language-aware separators.

    Uses ``RecursiveCharacterTextSplitter.from_language()`` which prioritises
    splitting on top-level boundaries (class definitions, function definitions,
    blank lines) before resorting to line or character splits.  This keeps
    functions and classes intact wherever the chunk size allows.

    Files with an unrecognised extension fall back to fixed chunking so that
    any file assigned to a ``codeAware`` group is still processed correctly.
    """
    text = getattr(doc, "text", "") or ""
    language = _detect_language(doc)

    if language is not None:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=conf.chunk_size,
            chunk_overlap=conf.overlap,
            length_function=length_fn,
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=conf.chunk_size,
            chunk_overlap=conf.overlap,
            length_function=length_fn,
        )

    return [ChunkDraft(text=t) for t in splitter.split_text(text)]
