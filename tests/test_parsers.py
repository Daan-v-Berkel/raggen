"""Tests for the parser layer."""
from __future__ import annotations

import pytest
from raggen.core.parsing.parser import ParseInput, ParserRegistry
from raggen.core.parsing.MarkdownParser import MarkdownParser
from raggen.core.parsing.PlainTextParser import PlainTextFallbackParser
from raggen.core.parsing.HtmlParser import HtmlParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inp(text: str, mimetype: str = "text/markdown", filename: str = "test.md") -> ParseInput:
    return ParseInput(
        doc_id="doc1",
        data=text.encode("utf-8"),
        mimetype=mimetype,
        filename=filename,
    )


MARKDOWN = """\
# Introduction

This is the intro.

## Installation

Run pip install raggen.

### Requirements

Python 3.10 or higher.

## Usage

raggen ingest
"""


# ---------------------------------------------------------------------------
# MarkdownParser
# ---------------------------------------------------------------------------

class TestMarkdownParser:
    def setup_method(self):
        self.parser = MarkdownParser()

    def test_parser_id(self):
        assert self.parser.parser_id == "markdown:v1"

    def test_supported_mimetypes(self):
        assert "text/markdown" in self.parser.supported_mimetypes
        assert "text/x-markdown" in self.parser.supported_mimetypes

    def test_heading_markers_preserved(self):
        result = self.parser.parse(_inp(MARKDOWN))
        assert "# Introduction" in result.document.text
        assert "## Installation" in result.document.text
        assert "### Requirements" in result.document.text

    def test_paragraph_breaks_preserved(self):
        result = self.parser.parse(_inp(MARKDOWN))
        # Double newlines between sections must survive
        assert "\n\n" in result.document.text

    def test_text_is_not_collapsed(self):
        """MarkdownParser must not run paragraph-merge normalization.
        Multiple consecutive blank lines should be preserved as-is."""
        md = "# Heading\n\n\n\nSome text."
        result = self.parser.parse(_inp(md))
        # Three newlines between heading and text must survive
        assert "\n\n\n" in result.document.text

    def test_doc_id_passed_through(self):
        result = self.parser.parse(_inp(MARKDOWN))
        assert result.document.doc_id == "doc1"

    def test_effective_mimetype(self):
        result = self.parser.parse(_inp(MARKDOWN))
        assert result.effective_mimetype == "text/markdown"

    def test_crlf_normalised(self):
        crlf = "# Heading\r\n\r\nSome text.\r\n"
        result = self.parser.parse(_inp(crlf))
        assert "\r" not in result.document.text
        assert "# Heading" in result.document.text

    def test_invalid_utf8_produces_warning(self):
        bad_bytes = b"# Heading\n\n" + bytes([0xFF, 0xFE]) + b" broken"
        inp = ParseInput(doc_id="bad", data=bad_bytes, mimetype="text/markdown", filename="bad.md")
        result = self.parser.parse(inp)
        assert result.encoding_error_ratio > 0
        assert len(result.warnings) == 1

    def test_empty_data_raises(self):
        inp = ParseInput(doc_id="x", data=b"", mimetype="text/markdown", filename="empty.md")
        with pytest.raises(ValueError, match="empty"):
            self.parser.parse(inp)

    def test_source_ref_set(self):
        result = self.parser.parse(_inp(MARKDOWN))
        assert result.document.source.rel_path == "test.md"
        assert result.document.source.scheme == "file"


# ---------------------------------------------------------------------------
# PlainTextFallbackParser
# ---------------------------------------------------------------------------

class TestPlainTextParser:
    def setup_method(self):
        self.parser = PlainTextFallbackParser()

    def test_parser_id(self):
        assert self.parser.parser_id == "plaintext:v1"

    def test_plain_text_round_trips(self):
        inp = ParseInput(doc_id="d", data=b"Hello world.", mimetype="text/plain", filename="f.txt")
        result = self.parser.parse(inp)
        assert result.document.text == "Hello world."

    def test_paragraphs_preserved(self):
        text = "First paragraph.\n\nSecond paragraph."
        inp = ParseInput(doc_id="d", data=text.encode(), mimetype="text/plain", filename="f.txt")
        result = self.parser.parse(inp)
        assert "First paragraph." in result.document.text
        assert "Second paragraph." in result.document.text

    def test_crlf_normalised(self):
        inp = ParseInput(doc_id="d", data=b"line1\r\nline2", mimetype="text/plain", filename="f.txt")
        result = self.parser.parse(inp)
        assert "\r" not in result.document.text

    def test_effective_mimetype_octet_stream_becomes_text_plain(self):
        inp = ParseInput(doc_id="d", data=b"text", mimetype="application/octet-stream", filename="f")
        result = self.parser.parse(inp)
        assert result.effective_mimetype == "text/plain"


# ---------------------------------------------------------------------------
# ParserRegistry dispatch
# ---------------------------------------------------------------------------

class TestParserRegistry:
    def setup_method(self):
        self.registry = ParserRegistry(fallback_parser=PlainTextFallbackParser())
        self.registry.register(MarkdownParser())

    def test_markdown_mimetype_routes_to_markdown_parser(self):
        parser = self.registry.resolve("text/markdown")
        assert isinstance(parser, MarkdownParser)

    def test_text_x_markdown_routes_to_markdown_parser(self):
        parser = self.registry.resolve("text/x-markdown")
        assert isinstance(parser, MarkdownParser)

    def test_plain_text_routes_to_fallback(self):
        parser = self.registry.resolve("text/plain")
        assert isinstance(parser, PlainTextFallbackParser)

    def test_unknown_mimetype_routes_to_fallback(self):
        parser = self.registry.resolve("application/unknown")
        assert isinstance(parser, PlainTextFallbackParser)

    def test_octet_stream_routes_to_fallback(self):
        parser = self.registry.resolve("application/octet-stream")
        assert isinstance(parser, PlainTextFallbackParser)

    def test_duplicate_mimetype_registration_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            self.registry.register(MarkdownParser())

    def test_markdown_parser_preserves_headings_end_to_end(self):
        """Full parse round-trip: .md bytes → Document with heading markers."""
        parser = self.registry.resolve("text/markdown")
        inp = _inp(MARKDOWN)
        doc = parser.parse(inp).document
        assert "# Introduction" in doc.text
        assert "## Installation" in doc.text

    def test_html_mimetype_routes_to_html_parser(self):
        registry = ParserRegistry(fallback_parser=PlainTextFallbackParser())
        registry.register(HtmlParser())
        assert isinstance(registry.resolve("text/html"), HtmlParser)

    def test_xhtml_mimetype_routes_to_html_parser(self):
        registry = ParserRegistry(fallback_parser=PlainTextFallbackParser())
        registry.register(HtmlParser())
        assert isinstance(registry.resolve("application/xhtml+xml"), HtmlParser)


# ---------------------------------------------------------------------------
# HtmlParser
# ---------------------------------------------------------------------------

def _html_inp(html: str, filename: str = "test.html") -> ParseInput:
    return ParseInput(
        doc_id="doc1",
        data=html.encode("utf-8"),
        mimetype="text/html",
        filename=filename,
    )


class TestHtmlParser:
    def setup_method(self):
        self.parser = HtmlParser()

    def test_parser_id(self):
        assert self.parser.parser_id == "html:v1"

    def test_supported_mimetypes(self):
        assert "text/html" in self.parser.supported_mimetypes
        assert "application/xhtml+xml" in self.parser.supported_mimetypes

    def test_h1_becomes_heading_marker(self):
        result = self.parser.parse(_html_inp("<h1>Title</h1><p>Body text.</p>"))
        assert "# Title" in result.document.text

    def test_h2_becomes_level_2_marker(self):
        result = self.parser.parse(_html_inp("<h2>Section</h2><p>Text.</p>"))
        assert "## Section" in result.document.text

    def test_h3_becomes_level_3_marker(self):
        result = self.parser.parse(_html_inp("<h3>Sub</h3><p>Text.</p>"))
        assert "### Sub" in result.document.text

    def test_h4_clamped_to_level_3(self):
        result = self.parser.parse(_html_inp("<h4>Deep</h4><p>Text.</p>"))
        assert "### Deep" in result.document.text
        assert "#### Deep" not in result.document.text

    def test_paragraph_text_included(self):
        result = self.parser.parse(_html_inp("<p>Hello world.</p>"))
        assert "Hello world." in result.document.text

    def test_script_tag_skipped(self):
        result = self.parser.parse(_html_inp(
            "<p>Visible.</p><script>alert('xss')</script>"
        ))
        assert "alert" not in result.document.text
        assert "Visible." in result.document.text

    def test_style_tag_skipped(self):
        result = self.parser.parse(_html_inp(
            "<style>body { color: red; }</style><p>Content.</p>"
        ))
        assert "color" not in result.document.text
        assert "Content." in result.document.text

    def test_pre_block_becomes_fenced_code(self):
        result = self.parser.parse(_html_inp(
            "<pre>def foo():\n    pass</pre>"
        ))
        assert "```" in result.document.text
        assert "def foo():" in result.document.text

    def test_nested_tags_text_preserved(self):
        result = self.parser.parse(_html_inp(
            "<p><strong>Bold</strong> and <em>italic</em> text.</p>"
        ))
        assert "Bold" in result.document.text
        assert "italic" in result.document.text

    def test_html_entities_decoded(self):
        result = self.parser.parse(_html_inp("<p>AT&amp;T &lt;rocks&gt;</p>"))
        assert "AT&T" in result.document.text
        assert "<rocks>" in result.document.text

    def test_full_page_heading_structure(self):
        html = """
        <html><body>
          <h1>Guide</h1>
          <p>Intro paragraph.</p>
          <h2>Installation</h2>
          <p>Run pip install.</p>
          <h3>Requirements</h3>
          <p>Python 3.10+</p>
        </body></html>
        """
        result = self.parser.parse(_html_inp(html))
        text = result.document.text
        assert "# Guide" in text
        assert "## Installation" in text
        assert "### Requirements" in text
        assert "Intro paragraph." in text

    def test_empty_data_raises(self):
        inp = ParseInput(doc_id="x", data=b"", mimetype="text/html", filename="e.html")
        with pytest.raises(ValueError, match="empty"):
            self.parser.parse(inp)

    def test_effective_mimetype(self):
        result = self.parser.parse(_html_inp("<p>hi</p>"))
        assert result.effective_mimetype == "text/html"
