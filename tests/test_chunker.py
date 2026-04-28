"""Tests for chunking strategies and the ChunkerRegistry."""
from __future__ import annotations

import pytest
from dataclasses import dataclass
from raggen.core.chunking.chunker import (
    _chunk_fixed,
    _chunk_paragraph,
    _chunk_heading,
    _chunk_code,
    _enrich_chunks,
    ChunkingError,
    StrategyChunker,
    ChunkerRegistry,
)
from raggen.core.chunking.chunks import ChunkDraft, Chunk
from raggen.core.config.project import GroupChunkingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeDoc:
    doc_id: str = "doc1"
    source: str = "docs/test.md"
    text: str = "Hello world."


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

class TestChunkerRegistryErrors:
    def test_unknown_strategy_raises(self):
        conf = _conf(strategy="nonexistent")
        # Bypass __init__ so we don't need a real ProjectConfig loaded
        registry = object.__new__(ChunkerRegistry)
        with pytest.raises(ChunkingError, match="Unknown chunking strategy"):
            registry._build_chunker(conf)


# ---------------------------------------------------------------------------
# StrategyChunker validates document
# ---------------------------------------------------------------------------

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
