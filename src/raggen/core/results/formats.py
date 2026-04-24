from __future__ import annotations

from enum import Enum


class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    XML = "xml"
