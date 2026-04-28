from typing import Protocol, Optional
from pydantic import BaseModel


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
    """
    The output contract for all parsers.

    ``text`` must always be Markdown-flavoured plain text:
      - headings expressed as ``#`` / ``##`` / ``###`` prefixes
      - paragraphs separated by ``\\n\\n``

    Parsers that handle structured formats (DOCX, HTML, …) are responsible
    for converting format-specific heading markers into this convention.
    Parsers for formats with no heading concept (plain text, code) simply
    return clean paragraph text.

    Chunking strategies read ``text`` directly and never need to know the
    original file format.
    """
    doc_id: str
    source: SourceRef
    text: str


class ParseResult(BaseModel):
    """Wrapper so the pipeline can introspect how parsing happened."""

    document: Document
    parser_id: str        # e.g. "markdown:v1", "plaintext:v1"
    effective_mimetype: str
    warnings: list[str] = []
    encoding_error_ratio: float = 0.0


class Parser(Protocol):
    """
    Interface every parser must satisfy.

    Implement ``parse()`` to convert raw bytes into a ``Document`` whose
    ``text`` follows the Markdown-flavoured convention described on
    ``Document``.  Register the parser with ``ParserRegistry`` for the
    mimetypes it handles.
    """
    parser_id: str
    supported_mimetypes: set[str]

    def parse(self, inp: ParseInput) -> ParseResult:
        ...


class ParserRegistry:
    def __init__(
        self, fallback_parser: "Parser", parsers_by_mime: dict[str, "Parser"] | None = None
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
    """Shared utility: normalise CRLF and bare CR to LF."""
    return s.replace("\r\n", "\n").replace("\r", "\n")
