from __future__ import annotations

from dataclasses import dataclass, field
from pydantic import BaseModel, NonNegativeInt
from typing import Literal, Optional, List

ChunkStrategy = Literal["fixed", "headingAware", "paragraphMerge", "codeAware"]
Unit = Literal["chars", "tokens"]


@dataclass
class ChunkDraft:
    """Intermediate output produced by a chunking strategy.

    Strategy functions return a list of ChunkDraft objects. _enrich_chunks
    then assembles them into full Chunk objects (adding IDs, config hashes,
    and stats). Strategies that only care about text can still return plain
    strings — _enrich_chunks accepts both.

    Fields
    ------
    text          : the chunk text (required)
    start_char    : character offset of the chunk's start in the source text
    end_char      : exclusive character offset of the chunk's end
    page_start    : first page number this chunk appears on (1-based)
    heading       : the immediate heading above this chunk
    section_path  : full heading hierarchy, e.g. ["Intro", "Installation"]
    """
    text: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    page_start: Optional[int] = None
    heading: Optional[str] = None
    section_path: Optional[List[str]] = field(default=None)


class Chunk(BaseModel):
    chunk_id: str  # = f"{doc_id}:{config_hash}:{chunk_index}"
    doc_id: str
    chunk_index: NonNegativeInt

    text: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None  # exclusive end

    class MetaData(BaseModel):
        page_start: Optional[int] = None
        page_end: Optional[int] = None
        heading: Optional[str] = None
        section_path: Optional[List[str]] = None
        source: Optional[object] = None

    metadata: MetaData

    class Stats(BaseModel):
        char_count: Optional[NonNegativeInt] = None
        token_count: Optional[NonNegativeInt] = None

    stats: Stats

    config_hash: str
