from pydantic import BaseModel, NonNegativeInt
from typing import Literal, Optional, List

ChunkStrategy = Literal["fixed", "headingAware",
                        "paragraphMerge", "tokenAware"]
Unit = Literal["chars", "tokens"]


class ChunkConfig(BaseModel):
    version: str = "v1"                      # bump if semantics change
    strategy: ChunkStrategy = "fixed"

    unit: Unit = "chars"                        # chars or tokens
    chunk_size: NonNegativeInt                    # max size in unit
    overlap: NonNegativeInt                       # overlap in unit

    # for recursive splitting / fallback boundaries
    separators: list[str] = ["\n\n", "\n", " "]
    preserve_newlines: bool = True            # if False, normalize whitespace

    # below this merge or drop (policy decides)
    min_chunk_size: NonNegativeInt
    merge_small_chunks: bool = True         # if True, merge tail fragments

    class IncludeMetaData(BaseModel):
        include_pages: bool = False
        include_headings: bool = False
        include_section_path: bool = False

    include_metadata: IncludeMetaData

    class Tokenizer(BaseModel):
        name: str
        version: str | None
        counting: Literal["exact", "approx"] = "exact"

    tokenizer: Tokenizer


class Chunk(BaseModel):
    chunk_id: str  # = f"{doc_id}:{config_hash}:{chunk_index}"
    doc_id: str
    chunk_index: NonNegativeInt

    text: str
    start_char: Optional[int] = None                    # offsets relative to *input text*, optional
    end_char: Optional[int] = None                      # exclusive end, optional

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

    config_hash: str                   # helps trace what produced it



DEFAULT_CHUNK_CONFIG = ChunkConfig(
    version="v1",
    strategy="fixed",

    # Use chars first for determinism + easy debugging
    unit="chars",
    chunk_size=1000,     # ~150–250 words depending on text; nice for UI + tests
    overlap=150,         # enough continuity without tons of duplication

    # Match your canonicalisation: paragraphs separated by "\n\n"
    separators=["\n\n", "\n", " "],
    preserve_newlines=True,

    # Avoid tiny junk chunks
    min_chunk_size=200,
    merge_small_chunks=False,

    include_metadata=ChunkConfig.IncludeMetaData(
        include_pages=False,
        include_headings=False,
        include_section_path=False,
    ),

    tokenizer=ChunkConfig.Tokenizer(
        name="none",          # not used when unit="chars"
        version=None,
        counting="exact",
    ),
)
