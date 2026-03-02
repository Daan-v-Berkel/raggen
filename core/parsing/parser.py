from typing import Dict, Protocol, List
from pydantic import BaseModel
from ..chunking.chunker import Document, OffsetRange
from dataclasses import dataclass
import re


class ParseInput(BaseModel):
    doc_id: str
    data: bytes
    mimetype: str
    filename: str | None


class ParseResult(BaseModel):
    """
    Wrapper so the pipeline can introspect how parsing happened without
    polluting Document for now.
    """
    document: Document
    parser_id: str                # e.g. "docx:v1", "plaintext:v1"
    effective_mimetype: str       # after normalization/fallback selection


class Parser(Protocol):
    parser_id: str
    supported_mimetypes: set[str]

    def parse(self, inp: ParseInput) -> ParseResult:
        pass


class UnsupportedDocumentError(ValueError):
    pass


class ParserRegistry:
    def __init__(self, fallback_parser, parsers_by_mime: dict[str, "Parser"] | None
                 = None):
        self.parsers_by_mime = parsers_by_mime or {}
        self.fallback_parser = fallback_parser

    def register(self, parser: "Parser") -> None:
        for mt in parser.supported_mimetypes:
            if mt in self.parsers_by_mime:
                raise ValueError(f"mimetype already registered: {mt}")
            self.parsers_by_mime[mt] = parser

    def resolve(self, mimetype: str) -> "Parser":
        if mimetype == "application/octet-stream":
            return self.fallback_parser
        return self.parsers_by_mime.get(mimetype, self.fallback_parser)


class ParserService:
    def __init__(self, registry: ParserRegistry):
        self.registry = registry

    def parse_document(self, inp: ParseInput) -> ParseResult:
        parser = self.registry.resolve(inp.mimetype)
        return parser.parse(inp)


@dataclass(frozen=True)
class _BuiltText:
    text: str
    paragraph_offsets: List[OffsetRange]


def _normalize_line_endings(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _split_paragraphs_drop_empty(s: str) -> List[str]:
    """
    Spec:
      - paragraphs separated by '\n\n'
      - drop empty paragraphs
      - keep paragraph ends (do not trim)
      - do not collapse spacing
    """
    if s == "":
        return []
    parts = re.split(r"\n{2,}", s)
    return [p for p in parts if p != ""]


def _build_canonical_text(paragraphs: List[str]) -> _BuiltText:
    """
    Join paragraphs with '\n\n' and compute deterministic offsets against final text.
    """
    buf: List[str] = []
    offsets: List[OffsetRange] = []
    cursor = 0

    for i, p in enumerate(paragraphs):
        if i > 0:
            sep = "\n\n"
            buf.append(sep)
            cursor += len(sep)

        start = cursor
        buf.append(p)
        cursor += len(p)
        end = cursor

        offsets.append(OffsetRange(start=start, end=end))

    return _BuiltText(text="".join(buf), paragraph_offsets=offsets)
