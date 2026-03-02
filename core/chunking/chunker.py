from .chunks import ChunkConfig, Chunk
from pydantic import BaseModel, NonNegativeInt
import hashlib
import json
from typing import List, Tuple, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter


class OffsetRange(BaseModel):
    start: NonNegativeInt
    end: NonNegativeInt


class Page(BaseModel):
    offset: OffsetRange
    pagenumber: NonNegativeInt


class HeadingBlock(BaseModel):
    offset: OffsetRange
    level: NonNegativeInt  # heading level 1..6
    text: str
    path: list[str]


class DocumentStructure(BaseModel):
    paragraphs: list[OffsetRange]
    headings: list[HeadingBlock]
    pages: list[Page]


class Document(BaseModel):
    structure: DocumentStructure
    doc_id: str
    text: str
    structure_version: str
    source: str | None = None


class ConfigError(ValueError):
    """
    Raised when a ChunkConfig is structurally valid (Pydantic),
    but semantically invalid for the Chunker (impossible combinations,
    unsupported strategy/structure mismatches, etc.).
    """

    def __init__(self, message: str, *, config: ChunkConfig | None = None):
        self.config = config
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        if self.config is not None:
            return f"{base}\nConfig: {self.config.model_dump()}"
        return base


class Chunker:

    def __init__(self, doc: Document):
        self.STRATEGIES = {
            "fixed": self._chunk_fixed,
            "headingAware": self._chunk_heading,
            "paragraphMerge": self._chunk_paragraph,
            "tokenAware": self._chunk_token,
        }
        self.document = doc

    def validate_config(self, conf: ChunkConfig) -> None:
        """
        Validate ChunkConfig in two phases:
          1) Pydantic validation (raises pydantic.ValidationError)
          2) "Impossible combo" validation (raises ConfigError with actionable messages)

        Returns: None (validation passes)
        """
        try:
            conf = ChunkConfig.model_validate(conf)
        except Exception:
            # Let Pydantic's own ValidationError bubble up unchanged
            raise

        errors: list[str] = []

        # Relationship constraints
        if conf.chunk_size > 0 and conf.overlap >= conf.chunk_size:
            errors.append(
                f"overlap ({conf.overlap}) must be smaller than chunk_size ({conf.chunk_size})."
            )

        if conf.min_chunk_size > 0 and conf.chunk_size > 0 and conf.min_chunk_size > conf.chunk_size:
            errors.append(
                f"min_chunk_size ({conf.min_chunk_size}) cannot be larger than chunk_size ({conf.chunk_size})."
            )

        # Unit/Tokenizer constraints
        if conf.unit == "tokens" and conf.tokenizer.name == "":
            errors.append(
                "unit='tokens' requires a tokenizer configuration.")

        if conf.include_metadata is not None:
            # If user requests page metadata but doc has no pages, fail fast (or switch to warning later)
            if conf.include_metadata.include_pages and len(self.document.structure.pages) == 0:
                errors.append(
                    "include_metadata.include_pages=True but Document.structure.pages is empty."
                )

        if conf.min_chunk_size > 0 and not conf.merge_small_chunks:
            # technically shouldn't be possible, drops chunks if chuns are smaller then min_chunk_size
            errors.append(
                "min_chunk_size is set but merge_small_chunks=False. Either enable merge_small_chunks "
                "or set min_chunk_size=0 to avoid ambiguous behavior."
            )

        if errors:
            raise ConfigError("Invalid ChunkConfig:\n- " + "\n- ".join(errors))

        return None

    def _stable_config_hash(self, conf: ChunkConfig) -> str:
        conf_json = json.dumps(
            conf.model_dump(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(conf_json.encode("utf-8")).hexdigest()

    def _find_offsets_sequential(self, text: str, pieces: List[str]) -> List[Tuple[int, int]]:
        """
        Deterministic offset finder:
        searches each chunk after the previous one, so repeated substrings don't break offsets.
        """
        offsets: List[Tuple[int, int]] = []
        cursor = 0
        for p in pieces:
            if not p:
                continue
            start = text.find(p, cursor)
            if start == -1:
                raise ValueError(
                    "Could not find chunk text in original document text (offset mapping failed).")
            end = start + len(p)
            offsets.append((start, end))
            cursor = max(cursor, end)  # move forward
        return offsets

    def _page_range_for_span(self, start: int, end: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Very simple mapping: find pages whose offsets intersect [start,end).
        """
        pages = [p for p in self.document.structure.pages if not (
            p.offset.end <= start or p.offset.start >= end)]
        if not pages:
            return None, None
        return pages[0].pagenumber, pages[-1].pagenumber

    def _heading_for_span(self, start: int, end: int) -> Tuple[Optional[str], Optional[list[str]]]:
        """
        Simplest mapping: pick the last heading whose start is before the chunk end.
        """
        candidates = [
            h for h in self.document.structure.headings if h.offset.start < end]
        if not candidates:
            return None, None
        h = candidates[-1]
        return h.text, h.path

    def _enrich_chunks(self, conf: ChunkConfig, pieces: list[str]) -> list[Chunk]:
        spans = self._find_offsets_sequential(self.document.text, pieces)

        config_hash = self._stable_config_hash(conf)

        out: List[Chunk] = []
        for idx, (piece, (start, end)) in enumerate(zip(pieces, spans)):
            page_start, page_end = self._page_range_for_span(
                start, end)
            heading, section_path = self._heading_for_span(
                start, end)

            chunk_id = f"{self.document.doc_id}:{config_hash}:{idx}"

            out.append(
                Chunk(
                    doc_id=self.document.doc_id,
                    chunk_index=idx,
                    text=piece,
                    start_char=start,
                    end_char=end,
                    metadata={
                        "page_start": page_start,
                        "page_end": page_end,
                        "heading": heading,
                        "section_path": section_path,
                        "source": getattr(self.document, "source", None),
                    },
                    stats={
                        "char_count": len(piece),
                        "token_count": None,  # TODO:add when adding token counting
                    },
                    config_hash=config_hash,
                    chunk_id=chunk_id,
                )
            )

        return out

    def chunk(self, conf: ChunkConfig) -> list[Chunk]:
        pieces = self.STRATEGIES[conf.strategy](conf)
        return self._enrich_chunks(conf, pieces)

    def _chunk_fixed(self, conf: ChunkConfig) -> list[Chunk]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=conf.chunk_size,
            chunk_overlap=conf.overlap,
            separators=conf.separators,
            keep_separator=conf.preserve_newlines,
        )
        return splitter.split_text(self.document.text)

    def _chunk_heading(self, conf: ChunkConfig) -> list[Chunk]:
        pass

    def _chunk_paragraph(self, conf: ChunkConfig) -> list[Chunk]:
        pass

    def _chunk_token(self, conf: ChunkConfig) -> list[Chunk]:
        pass
