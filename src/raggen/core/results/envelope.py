from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class ResultMessage(BaseModel):
    code: str
    message: str


class ResultMeta(BaseModel):
    trace_id: str | None = None
    duration_ms: int | None = None


class ResultEnvelope(BaseModel):
    schema_version: str = "1"
    operation: str
    success: bool

    # Operation-specific payload.
    data: Any = None

    warnings: list[ResultMessage] = Field(default_factory=list)
    errors: list[ResultMessage] = Field(default_factory=list)
    meta: Optional[ResultMeta] = None

    def to_plain(self) -> dict[str, Any]:
        """
        Convert to a canonical plain Python structure that renderers can consume.
        """
        return self.model_dump(mode="json", exclude_none=True)
