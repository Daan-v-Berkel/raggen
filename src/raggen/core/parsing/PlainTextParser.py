import re
from pathlib import Path

from .parser import (
    Document,
    ParseInput,
    ParseResult,
    SourceRef,
    _normalize_line_endings,
)


def _split_paragraphs_drop_empty(s: str) -> list[str]:
    """Split on two-or-more consecutive newlines; drop empty segments."""
    if not s:
        return []
    return [p for p in re.split(r"\n{2,}", s) if p]


class PlainTextFallbackParser:
    """
    Fallback parser for plain text and unrecognised file types.

    Decodes bytes, normalises line endings, collapses multiple blank lines
    into a single paragraph break (``\\n\\n``), and returns clean paragraph
    text.  No heading markers are added — the file is treated as unstructured
    prose.  Chunking strategies that need heading structure (e.g.
    ``headingAware``) will find no ``#`` markers and degrade gracefully to
    paragraph-level splitting.
    """

    parser_id: str = "plaintext:v1"
    supported_mimetypes: set[str] = {"text/plain"}

    def parse(self, inp: ParseInput) -> ParseResult:
        if not inp.data:
            raise ValueError("ParseInput.data is empty")

        warnings: list[str] = []
        encoding_error_ratio = 0.0

        try:
            raw = inp.data.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raw = inp.data.decode("utf-8", errors="replace")
            replacement_count = raw.count("�")
            encoding_error_ratio = replacement_count / max(len(raw), 1)
            warnings.append(
                f"{inp.doc_id}: file contains invalid UTF-8 bytes "
                f"({replacement_count} replacement characters, "
                f"{encoding_error_ratio:.1%} of content)."
            )

        raw = _normalize_line_endings(raw)
        paragraphs = _split_paragraphs_drop_empty(raw)
        text = "\n\n".join(paragraphs)

        src = SourceRef(
            scheme="file",
            rel_path=getattr(inp, "filename", None) or inp.doc_id,
            display_name=Path(getattr(inp, "filename", inp.doc_id)).name,
        )

        doc = Document(
            doc_id=inp.doc_id,
            text=text,
            source=src,
        )

        effective = (
            "text/plain" if inp.mimetype == "application/octet-stream" else inp.mimetype
        )

        return ParseResult(
            document=doc,
            parser_id=self.parser_id,
            effective_mimetype=effective,
            warnings=warnings,
            encoding_error_ratio=encoding_error_ratio,
        )
