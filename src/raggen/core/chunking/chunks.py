from pydantic import BaseModel, NonNegativeInt
from typing import Literal, Optional, List

ChunkStrategy = Literal["fixed", "headingAware", "paragraphMerge", "tokenAware"]
Unit = Literal["chars", "tokens"]


class Chunk(BaseModel):
    chunk_id: str  # = f"{doc_id}:{config_hash}:{chunk_index}"
    doc_id: str
    chunk_index: NonNegativeInt

    text: str
    # offsets relative to *input text*, optional
    start_char: Optional[int] = None
    # exclusive end, optional
    end_char: Optional[int] = None

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

    config_hash: str  # helps trace what produced it
