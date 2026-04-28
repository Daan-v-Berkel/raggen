from html.parser import HTMLParser as _StdlibHTMLParser
from pathlib import Path

from .parser import (
    Document,
    ParseInput,
    ParseResult,
    SourceRef,
    _normalize_line_endings,
)

# Tags whose content must be silently skipped (not user-visible text)
_SKIP_TAGS = {"script", "style", "head", "noscript", "meta", "link", "svg"}

# h1-h6 -> heading level integer
_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

# Block-level tags that act as paragraph separators
_BLOCK_TAGS = {
    "p", "div", "section", "article", "main", "header", "footer",
    "nav", "aside", "blockquote", "li", "ul", "ol",
    "table", "tr", "td", "th", "dd", "dt", "figure", "figcaption",
}


class _ToMarkdown(_StdlibHTMLParser):
    """
    Minimal HTML → Markdown-flavoured text converter.

    Rules:
    - <h1>–<h6>  → # through ### (clamped to 3 levels)
    - <p>, block tags → paragraph break
    - <pre>       → fenced code block
    - <br>        → newline
    - <hr>        → ---
    - <script>, <style>, <head>, … → skipped entirely
    - Everything else → plain text content only
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._buf: list[str] = []
        self._skip: int = 0    # depth inside a skip-tag subtree
        self._heading: int = 0 # current heading level, 0 = not in heading
        self._pre: int = 0     # depth inside <pre>

    # --- SAX callbacks ---------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip += 1
            return
        if self._skip:
            return

        if tag == "pre":
            self._commit()
            self._pre += 1
            self._buf.append("```\n")
        elif tag in _HEADING_TAGS:
            self._commit()
            self._heading = _HEADING_TAGS[tag]
        elif tag in _BLOCK_TAGS:
            self._commit()
        elif tag == "br" and not self._pre:
            self._buf.append("\n")
        elif tag == "hr":
            self._commit()
            self._parts.append("---\n\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return

        if tag == "pre":
            # Emit everything accumulated inside <pre> as a fenced block
            text = "".join(self._buf)
            if text and not text.endswith("\n"):
                text += "\n"
            self._parts.append(text + "```\n\n")
            self._buf = []
            self._pre = max(0, self._pre - 1)
        elif tag in _HEADING_TAGS:
            self._commit()
        elif tag in _BLOCK_TAGS:
            self._commit()

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        self._buf.append(data)

    # --- Internal --------------------------------------------------------

    def _commit(self) -> None:
        """Flush the current buffer as a heading or paragraph."""
        if self._pre:
            # Inside <pre> content is handled by handle_endtag("pre")
            return
        text = "".join(self._buf).strip()
        self._buf = []
        if not text:
            self._heading = 0
            return
        if self._heading:
            level = min(self._heading, 3)  # clamp: our chunker only tracks H1-H3
            self._parts.append("#" * level + " " + text + "\n\n")
            self._heading = 0
        else:
            self._parts.append(text + "\n\n")

    def result(self) -> str:
        self._commit()
        return "".join(self._parts).strip()


def _html_to_markdown(html: str) -> str:
    converter = _ToMarkdown()
    converter.feed(html)
    return converter.result()


class HtmlParser:
    """
    Parser for HTML files (.html, .htm).

    Converts HTML to Markdown-flavoured text:
    - Heading tags become # / ## / ### markers
    - Block elements become paragraph breaks
    - <pre> blocks become fenced code blocks
    - <script>, <style>, <head> are discarded

    Uses only Python stdlib (html.parser) — no extra dependencies.
    """

    parser_id: str = "html:v1"
    supported_mimetypes: set[str] = {"text/html", "application/xhtml+xml"}

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
        text = _html_to_markdown(raw)

        src = SourceRef(
            scheme="file",
            rel_path=getattr(inp, "filename", None) or inp.doc_id,
            display_name=Path(getattr(inp, "filename", inp.doc_id)).name,
        )

        return ParseResult(
            document=Document(doc_id=inp.doc_id, text=text, source=src),
            parser_id=self.parser_id,
            effective_mimetype="text/html",
            warnings=warnings,
            encoding_error_ratio=encoding_error_ratio,
        )
