from typing import Protocol, List, Optional
from pydantic import BaseModel
from dataclasses import dataclass
import re


class ParseInput(BaseModel):
    doc_id: str
    data: bytes
    mimetype: str
    filename: str


class SourceRef(BaseModel):
    scheme: str = "file"
    rel_path: str
    display_name: Optional[str] = None


class Document(BaseModel):
    doc_id: str
    source: SourceRef
    text: str


class ParseResult(BaseModel):
    """
    Wrapper so the pipeline can introspect how parsing happened.
    document: Document produced by the parser
    """

    document: Document
    parser_id: str  # e.g. "docx:v1", "plaintext:v1"
    effective_mimetype: str  # after normalization/fallback selection
    warnings: list[str] = []
    encoding_error_ratio: float = 0.0


class Parser(Protocol):
    parser_id: str
    supported_mimetypes: set[str]

    def parse(self, inp: ParseInput) -> ParseResult:
        pass


class UnsupportedDocumentError(ValueError):
    pass


class ParserRegistry:
    def __init__(
        self, fallback_parser, parsers_by_mime: dict[str, "Parser"] | None = None
    ):
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


@dataclass(frozen=True)
class _BuiltText:
    text: str
    paragraphs: List[str]


def _build_canonical_text(paragraphs: List[str]) -> _BuiltText:
    """
    Join paragraphs with '\n\n' and return the canonical text and paragraph list.
    """
    buf: List[str] = []
    for i, p in enumerate(paragraphs):
        if i > 0:
            buf.append("\n\n")
        buf.append(p)
    return _BuiltText(text="".join(buf), paragraphs=paragraphs)
