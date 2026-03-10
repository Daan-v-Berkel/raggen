from .parser import (
    ParseResult,
    _normalize_line_endings,
    _build_canonical_text,
    _split_paragraphs_drop_empty,
    Document,
    SourceRef,
)
from pathlib import Path


class PlainTextFallbackParser:
    """
    Fallback parser for simple/unknown files.

    - Input: bytes + mimetype (required)
    - Output: canonical Document according to the canonicalisation spec
    - Headings: none (v1)
    - Pages: single page spanning whole text (v1)
    """

    parser_id: str = "plaintext:v1"
    supported_mimetypes: set[str] = {"text/plain", "application/octet-stream"}

    def parse(self, inp) -> "ParseResult":
        if not inp.data:
            raise ValueError("ParseInput.data is empty")

        raw = inp.data.decode("utf-8", errors="replace")

        raw = _normalize_line_endings(raw)
        paragraphs = _split_paragraphs_drop_empty(raw)
        built = _build_canonical_text(paragraphs)

        # Build Document with simple SourceRef
        src = SourceRef(
            scheme="file",
            rel_path=getattr(inp, "filename", None) or inp.doc_id,
            display_name=Path(getattr(inp, "filename", inp.doc_id)).name,
        )

        doc = Document(
            doc_id=inp.doc_id,
            text=built.text,
            source=src,
        )

        effective = (
            "text/plain" if inp.mimetype == "application/octet-stream" else inp.mimetype
        )

        return ParseResult(
            document=doc,
            parser_id=self.parser_id,
            effective_mimetype=effective,
        )
