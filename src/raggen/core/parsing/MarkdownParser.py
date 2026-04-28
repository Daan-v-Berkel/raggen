from pathlib import Path

from .parser import (
    Document,
    ParseInput,
    ParseResult,
    SourceRef,
    _normalize_line_endings,
)


class MarkdownParser:
    """
    Parser for Markdown files (.md, .markdown).

    Responsibility: decode bytes and normalise line endings, then return the
    text as-is.  Markdown headings (# / ## / ###) and paragraph breaks (\n\n)
    are already the canonical structure that chunking strategies such as
    ``headingAware`` rely on — stripping or re-joining them here would destroy
    that signal.

    No extra dependencies required.
    """

    parser_id: str = "markdown:v1"
    supported_mimetypes: set[str] = {"text/markdown", "text/x-markdown"}

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

        text = _normalize_line_endings(raw)

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

        return ParseResult(
            document=doc,
            parser_id=self.parser_id,
            effective_mimetype="text/markdown",
            warnings=warnings,
            encoding_error_ratio=encoding_error_ratio,
        )
