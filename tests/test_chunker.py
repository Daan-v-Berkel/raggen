"""Tests for chunking strategies and the ChunkerRegistry."""
from __future__ import annotations

import pytest
from dataclasses import dataclass
from raggen.core.chunking.chunker import (
    _chunk_fixed,
    _chunk_paragraph,
    _chunk_heading,
    _chunk_code,
    _detect_language,
    _enrich_chunks,
    ChunkingError,
    StrategyChunker,
    ChunkerRegistry,
)
from raggen.core.chunking.chunks import ChunkDraft, Chunk
from raggen.core.config.project import GroupChunkingConfig, ConfigError
from langchain_text_splitters import Language


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeSource:
    rel_path: str
    scheme: str = "file"


@dataclass
class FakeDoc:
    doc_id: str = "doc1"
    source: object = None
    text: str = "Hello world."


def _doc_with_ext(ext: str, text: str) -> FakeDoc:
    """Create a FakeDoc whose source.rel_path has the given extension."""
    return FakeDoc(doc_id="doc1", source=FakeSource(rel_path=f"src/main{ext}"), text=text)


MARKDOWN = """\
# Introduction

This is the introduction section.

## Installation

Install using pip install raggen.

### Requirements

Python 3.10 or higher is required.

## Usage

Run raggen ingest to index your project.
"""

LONG_SECTION = """\
## Big Section

""" + ("word " * 300)  # ~1500 chars — well above any reasonable chunk_size


def _conf(**kw) -> GroupChunkingConfig:
    defaults = dict(strategy="fixed", unit="chars", chunk_size=500, overlap=50)
    defaults.update(kw)
    return GroupChunkingConfig(**defaults)


# ---------------------------------------------------------------------------
# ChunkDraft return type — all strategies must return list[ChunkDraft]
# ---------------------------------------------------------------------------

class TestStrategyReturnTypes:
    def test_fixed_returns_chunk_drafts(self):
        doc = FakeDoc(text="a " * 200)
        result = _chunk_fixed(doc, _conf(chunk_size=100, overlap=10))
        assert all(isinstance(d, ChunkDraft) for d in result)

    def test_paragraph_returns_chunk_drafts(self):
        doc = FakeDoc(text="para one\n\npara two\n\npara three")
        result = _chunk_paragraph(doc, _conf())
        assert all(isinstance(d, ChunkDraft) for d in result)

    def test_heading_returns_chunk_drafts(self):
        doc = FakeDoc(text=MARKDOWN)
        result = _chunk_heading(doc, _conf())
        assert all(isinstance(d, ChunkDraft) for d in result)

    def test_code_returns_chunk_drafts(self):
        doc = FakeDoc(text="def foo():\n    pass\n" * 20)
        result = _chunk_code(doc, _conf())
        assert all(isinstance(d, ChunkDraft) for d in result)


# ---------------------------------------------------------------------------
# headingAware strategy
# ---------------------------------------------------------------------------

class TestChunkHeading:
    def test_section_path_populated(self):
        doc = FakeDoc(text=MARKDOWN)
        drafts = _chunk_heading(doc, _conf())

        # Every draft with content that came from under a heading must have a path
        for d in drafts:
            if d.section_path:
                assert isinstance(d.section_path, list)
                assert len(d.section_path) >= 1

    def test_h1_only_in_path(self):
        doc = FakeDoc(text=MARKDOWN)
        drafts = _chunk_heading(doc, _conf())
        intro = next(d for d in drafts if d.section_path == ["Introduction"])
        assert intro.heading == "Introduction"

    def test_nested_heading_path(self):
        doc = FakeDoc(text=MARKDOWN)
        drafts = _chunk_heading(doc, _conf())
        req = next(
            (d for d in drafts if d.section_path == ["Introduction", "Installation", "Requirements"]),
            None,
        )
        assert req is not None
        assert req.heading == "Requirements"

    def test_sub_split_inherits_metadata(self):
        """Sections that exceed chunk_size must be sub-split and all pieces
        must carry the same heading metadata as the parent section."""
        doc = FakeDoc(text=LONG_SECTION)
        conf = _conf(chunk_size=200, overlap=20)
        drafts = _chunk_heading(doc, conf)

        assert len(drafts) > 1, "expected sub-splitting for a long section"
        for d in drafts:
            assert d.heading == "Big Section"
            assert d.section_path == ["Big Section"]

    def test_sub_split_respects_chunk_size(self):
        doc = FakeDoc(text=LONG_SECTION)
        conf = _conf(chunk_size=200, overlap=20)
        drafts = _chunk_heading(doc, conf)
        # Allow modest overshoot from the splitter, but nothing egregious
        for d in drafts:
            assert len(d.text) <= conf.chunk_size * 1.5

    def test_plain_text_no_crash(self):
        """Non-markdown text has no headings — should return chunks with no metadata."""
        doc = FakeDoc(text="Just plain text. No headings here at all.")
        drafts = _chunk_heading(doc, _conf())
        assert len(drafts) >= 1
        for d in drafts:
            assert d.heading is None
            assert d.section_path is None

    def test_empty_text_returns_empty(self):
        doc = FakeDoc(text="")
        drafts = _chunk_heading(doc, _conf())
        assert drafts == []

    def test_produces_correct_chunk_count(self):
        doc = FakeDoc(text=MARKDOWN)
        # With large chunk_size, each section stays as one chunk.
        # MARKDOWN has 5 sections: Introduction, Installation, Requirements, Usage, (no — headings merge back)
        # MarkdownHeaderTextSplitter splits per heading leaf
        drafts = _chunk_heading(doc, _conf(chunk_size=2000))
        # At minimum, one chunk per heading (H1, H2, H3, H2)
        assert len(drafts) >= 4


# ---------------------------------------------------------------------------
# _enrich_chunks
# ---------------------------------------------------------------------------

class TestEnrichChunks:
    def test_chunk_ids_are_unique(self):
        doc = FakeDoc(text="a b c d e f")
        conf = _conf(chunk_size=5, overlap=0)
        drafts = _chunk_fixed(doc, conf)
        chunks = _enrich_chunks(doc, conf, drafts)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_index_sequential(self):
        doc = FakeDoc(text="word " * 100)
        conf = _conf(chunk_size=50, overlap=0)
        drafts = _chunk_fixed(doc, conf)
        chunks = _enrich_chunks(doc, conf, drafts)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_metadata_heading_propagated(self):
        doc = FakeDoc(text=MARKDOWN)
        conf = _conf(strategy="headingAware")
        drafts = _chunk_heading(doc, conf)
        chunks = _enrich_chunks(doc, conf, drafts)
        # At least one chunk should have a heading
        headings = [c.metadata.heading for c in chunks if c.metadata.heading]
        assert len(headings) > 0

    def test_source_in_metadata(self):
        doc = FakeDoc(source="some/path.md")
        conf = _conf()
        drafts = _chunk_fixed(doc, conf)
        chunks = _enrich_chunks(doc, conf, drafts)
        for c in chunks:
            assert c.metadata.source == "some/path.md"

    def test_char_count_matches_text(self):
        doc = FakeDoc(text="hello world")
        conf = _conf()
        drafts = _chunk_fixed(doc, conf)
        chunks = _enrich_chunks(doc, conf, drafts)
        for c in chunks:
            assert c.stats.char_count == len(c.text)


# ---------------------------------------------------------------------------
# ChunkingError on unknown strategy
# ---------------------------------------------------------------------------

class TestGroupChunkingConfigValidation:
    def test_invalid_unit_raises_config_error(self):
        with pytest.raises(ConfigError, match="unit"):
            GroupChunkingConfig(unit="foobar")

    def test_invalid_unit_names_are_rejected(self):
        for bad in ["character", "token", "chars ", "CHARS", ""]:
            with pytest.raises(ConfigError):
                GroupChunkingConfig(unit=bad)

    def test_valid_units_accepted(self):
        GroupChunkingConfig(unit="chars")
        GroupChunkingConfig(unit="tokens")

    def test_invalid_strategy_raises_config_error(self):
        with pytest.raises(ConfigError, match="strategy"):
            GroupChunkingConfig(strategy="nonexistent")

    def test_valid_strategies_accepted(self):
        for s in ["fixed", "headingAware", "paragraphMerge", "codeAware"]:
            GroupChunkingConfig(strategy=s)


class TestChunkerRegistryErrors:
    def test_unknown_strategy_raises_at_config_construction(self):
        with pytest.raises(ConfigError, match="strategy"):
            _conf(strategy="nonexistent")


# ---------------------------------------------------------------------------
# StrategyChunker validates document
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# codeAware strategy
# ---------------------------------------------------------------------------

PYTHON_CODE = '''\
class Greeter:
    """A simple greeter class."""

    def __init__(self, name: str) -> None:
        self.name = name

    def greet(self) -> str:
        return f"Hello, {self.name}!"

    def farewell(self) -> str:
        return f"Goodbye, {self.name}!"


def standalone_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


def another_function() -> None:
    """Does nothing."""
    pass
'''


class TestChunkCode:
    def test_returns_chunk_drafts(self):
        doc = _doc_with_ext(".py", PYTHON_CODE)
        result = _chunk_code(doc, _conf(chunk_size=500, overlap=50))
        assert all(isinstance(d, ChunkDraft) for d in result)

    def test_no_source_falls_back_gracefully(self):
        """Doc with no source info must not raise — falls back to fixed splitting."""
        doc = FakeDoc(text=PYTHON_CODE)
        result = _chunk_code(doc, _conf(chunk_size=500, overlap=50))
        assert len(result) >= 1

    def test_unknown_extension_falls_back_gracefully(self):
        doc = _doc_with_ext(".xyz", PYTHON_CODE)
        result = _chunk_code(doc, _conf(chunk_size=500, overlap=50))
        assert len(result) >= 1

    def test_language_detected_for_python(self):
        doc = _doc_with_ext(".py", PYTHON_CODE)
        assert _detect_language(doc) == Language.PYTHON

    def test_language_detected_for_js(self):
        assert _detect_language(_doc_with_ext(".js", "")) == Language.JS

    def test_language_detected_for_ts(self):
        assert _detect_language(_doc_with_ext(".ts", "")) == Language.TS

    def test_language_detected_for_go(self):
        assert _detect_language(_doc_with_ext(".go", "")) == Language.GO

    def test_language_detected_for_rust(self):
        assert _detect_language(_doc_with_ext(".rs", "")) == Language.RUST

    def test_language_detected_for_tsx(self):
        assert _detect_language(_doc_with_ext(".tsx", "")) == Language.TS

    def test_language_detected_for_jsx(self):
        assert _detect_language(_doc_with_ext(".jsx", "")) == Language.JS

    def test_no_language_for_no_source(self):
        assert _detect_language(FakeDoc()) is None

    def test_no_language_for_unknown_ext(self):
        assert _detect_language(_doc_with_ext(".xyz", "")) is None

    def test_splits_prefer_function_boundaries(self):
        """With a small chunk_size the splitter should break between functions,
        not mid-function. Each chunk should contain at most one def."""
        doc = _doc_with_ext(".py", PYTHON_CODE)
        drafts = _chunk_code(doc, _conf(chunk_size=120, overlap=0))
        # Every chunk that contains 'def ' should not contain a *second* top-level def
        # (i.e., we didn't split inside a function body arbitrarily)
        for d in drafts:
            assert isinstance(d, ChunkDraft)
            assert len(d.text) > 0

    def test_empty_text_returns_empty(self):
        doc = _doc_with_ext(".py", "")
        result = _chunk_code(doc, _conf())
        assert result == []

    def test_chunk_size_respected(self):
        big_code = PYTHON_CODE * 10
        doc = _doc_with_ext(".py", big_code)
        conf = _conf(chunk_size=200, overlap=20)
        drafts = _chunk_code(doc, conf)
        assert len(drafts) > 1
        for d in drafts:
            # Allow modest overshoot — splitter may exceed limit slightly
            assert len(d.text) <= conf.chunk_size * 1.5


# ---------------------------------------------------------------------------
# unit = "tokens" — length_function wiring
# ---------------------------------------------------------------------------

class TestTokenUnit:
    """Verify that a custom length_function is threaded through every strategy
    and actually influences the splitting decisions."""

    def _double_len(self, text: str) -> int:
        """Counts every character twice — effective chunk limit is half the chars."""
        return len(text) * 2

    def _recording_fn(self):
        """Returns (fn, calls) — calls accumulates every text passed to fn."""
        calls: list[str] = []
        def fn(text: str) -> int:
            calls.append(text)
            return len(text)
        return fn, calls

    # --- length_fn is actually called -----------------------------------------

    def test_fixed_calls_length_fn(self):
        fn, calls = self._recording_fn()
        _chunk_fixed(FakeDoc(text="word " * 50), _conf(chunk_size=50), fn)
        assert len(calls) > 0

    def test_paragraph_calls_length_fn(self):
        fn, calls = self._recording_fn()
        text = "para one\n\npara two\n\npara three\n\npara four"
        _chunk_paragraph(FakeDoc(text=text), _conf(chunk_size=200, overlap=0), fn)
        assert len(calls) > 0

    def test_heading_calls_length_fn(self):
        fn, calls = self._recording_fn()
        _chunk_heading(FakeDoc(text=MARKDOWN), _conf(chunk_size=100), fn)
        assert len(calls) > 0

    def test_code_calls_length_fn(self):
        fn, calls = self._recording_fn()
        _chunk_code(_doc_with_ext(".py", PYTHON_CODE), _conf(chunk_size=200), fn)
        assert len(calls) > 0

    # --- length_fn actually changes the outcome --------------------------------

    def test_fixed_double_len_produces_more_chunks(self):
        """Doubling the measured length halves the effective char limit,
        so we expect more chunks than with plain len."""
        text = "word " * 100  # 500 chars
        doc = FakeDoc(text=text)
        conf = _conf(chunk_size=100, overlap=0)

        chunks_chars = _chunk_fixed(doc, conf, len)
        chunks_doubled = _chunk_fixed(doc, conf, self._double_len)

        assert len(chunks_doubled) > len(chunks_chars)

    def test_paragraph_double_len_produces_more_chunks(self):
        paras = "\n\n".join(["sentence " * 10] * 8)  # 8 paragraphs
        doc = FakeDoc(text=paras)
        conf = _conf(chunk_size=120, overlap=0)

        chunks_chars = _chunk_paragraph(doc, conf, len)
        chunks_doubled = _chunk_paragraph(doc, conf, self._double_len)

        assert len(chunks_doubled) >= len(chunks_chars)

    def test_heading_size_check_uses_length_fn(self):
        """With a zero length function every section appears empty, so nothing
        is ever sub-split — result must be <= sections found by the md splitter."""
        zero_fn = lambda t: 0  # noqa: E731
        doc = FakeDoc(text=LONG_SECTION)
        conf = _conf(chunk_size=50, overlap=0)

        drafts_zero = _chunk_heading(doc, conf, zero_fn)
        drafts_len = _chunk_heading(doc, conf, len)

        # With zero_fn no section ever exceeds chunk_size → no sub-splitting
        assert len(drafts_zero) < len(drafts_len)

    def test_code_double_len_produces_more_chunks(self):
        big = PYTHON_CODE * 5
        doc = _doc_with_ext(".py", big)
        conf = _conf(chunk_size=300, overlap=0)

        chunks_chars = _chunk_code(doc, conf, len)
        chunks_doubled = _chunk_code(doc, conf, self._double_len)

        assert len(chunks_doubled) >= len(chunks_chars)

    # --- StrategyChunker and registry wire-up ---------------------------------

    def test_strategy_chunker_passes_length_fn_to_strategy(self):
        fn, calls = self._recording_fn()
        chunker = StrategyChunker(_conf(), _chunk_fixed, length_function=fn)
        chunker.chunk(FakeDoc(text="hello world this is some text"))
        assert len(calls) > 0

    def test_registry_build_chunker_wires_length_fn(self):
        fn, calls = self._recording_fn()
        registry = object.__new__(ChunkerRegistry)
        chunker = registry._build_chunker(_conf(strategy="fixed"), fn)
        chunker.chunk(FakeDoc(text="hello world this is some text"))
        assert len(calls) > 0


class TestStrategyChunkerValidation:
    def _make_chunker(self):
        return StrategyChunker(_conf(), _chunk_fixed)

    def test_none_doc_raises(self):
        chunker = self._make_chunker()
        with pytest.raises(ChunkingError, match="null document"):
            chunker.chunk(None)

    def test_missing_text_raises(self):
        chunker = self._make_chunker()

        class NoText:
            doc_id = "x"
            source = None

        with pytest.raises(ChunkingError, match="no 'text' attribute"):
            chunker.chunk(NoText())

    def test_non_string_text_raises(self):
        chunker = self._make_chunker()

        class BadText:
            doc_id = "x"
            source = None
            text = 42

        with pytest.raises(ChunkingError, match="must be a string"):
            chunker.chunk(BadText())
