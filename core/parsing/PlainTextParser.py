from .parser import ParseResult, _normalize_line_endings, _build_canonical_text, _split_paragraphs_drop_empty, _BuiltText

from ..chunking.chunker import Document, DocumentStructure, OffsetRange, Page


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

        # Minimal structure (v1)
        structure = DocumentStructure(
            paragraphs=built.paragraph_offsets,
            headings=[],
            pages=[
                Page(
                    offset=OffsetRange(start=0, end=len(built.text)),
                    pagenumber=1,
                )
            ],
        )

        doc = Document(
            doc_id=inp.doc_id,
            text=built.text,
            structure=structure,
            structure_version="canon:1.0|plaintext:v1",
            source=getattr(inp, "filename", None) or inp.doc_id,
        )

        effective = "text/plain" if inp.mimetype == "application/octet-stream" else inp.mimetype

        return ParseResult(
            document=doc,
            parser_id=self.parser_id,
            effective_mimetype=effective,
        )
