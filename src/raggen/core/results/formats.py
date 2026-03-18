from __future__ import annotations

from enum import Enum


class OutputFormat(str, Enum):
    JSON = "json"
    XML = "xml"
