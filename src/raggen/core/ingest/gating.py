from __future__ import annotations

from typing import Tuple, Optional


def should_ingest_raw_bytes(data: bytes) -> Tuple[bool, Optional[str]]:
    """Decide whether raw bytes should be ingested.

    Returns (True, None) to proceed, or (False, reason) to skip.
    """
    if data is None:
        return False, "empty_bytes"
    if len(data) == 0:
        return False, "empty_bytes"
    return True, None


def should_ingest_parsed_document(doc) -> Tuple[bool, Optional[str]]:
    """Decide whether a parsed Document should be ingested.

    Expects `doc` to have a `text` attribute (string) or similar.
    Returns (False, "empty_text_after_parse") when text is empty or whitespace.
    """
    if doc is None:
        return False, "empty_text_after_parse"
    text = getattr(doc, "text", None)
    if text is None:
        return False, "empty_text_after_parse"
    try:
        if str(text).strip() == "":
            return False, "empty_text_after_parse"
    except Exception:
        return False, "empty_text_after_parse"
    return True, None
