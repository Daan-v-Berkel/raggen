from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from uuid import uuid4


class ResultMessage(BaseModel):
    code: str
    message: str


class ResultMeta(BaseModel):
    trace_id: str | None = None
    duration_ms: int | None = None


class ResultEnvelope(BaseModel):
    schema_version: str = "1"
    run_id: str
    created_at: str
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


def init_result(operation: str) -> ResultEnvelope:
    now = datetime.now(timezone.utc)
    run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}_{operation}_{uuid4().hex[:6]}"

    return ResultEnvelope(
        run_id=run_id,
        created_at=now.isoformat(),
        operation=operation,
        success=False,
    )
