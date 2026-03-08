from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationResult:
    text: str
    model_id: str
    provider: str
    usage: Optional[dict] = None
