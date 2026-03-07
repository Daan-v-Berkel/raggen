from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QueryRequest:
    text: str
    top_k: int = 8
    generate_answer: bool = False

    # Optional overrides
    query_model_id: Optional[str] = None
    llm_model_id: Optional[str] = None

    # Future extension point
    filters: dict | None = None
    max_context_chunks: Optional[int] = None

    def __post_init__(self) -> None:
        self.text = self.text.strip()
        if not self.text:
            raise ValueError("Query text must not be empty.")
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0.")
        if self.max_context_chunks is not None and self.max_context_chunks <= 0:
            raise ValueError("max_context_chunks must be > 0 when provided.")


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    chunk_index: int

    # Optional metadata
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    page_number: Optional[int] = None
    heading_path: Optional[list[str]] = None


@dataclass
class QueryResponse:
    query: str
    matches: list[RetrievedChunk] = field(default_factory=list)
    answer: Optional[str] = None

    used_query_model: str = ""
    used_llm_model: Optional[str] = None
