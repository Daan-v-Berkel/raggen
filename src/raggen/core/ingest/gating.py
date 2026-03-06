from __future__ import annotations

from sqlalchemy import select
from typing import Tuple, Optional
from raggen.core.store.metadata_schema import documents
from raggen.core.scanner import FileRef
from raggen.core.store.engine import create_engine_from_url
from raggen.core.config.project import ProjectConfig


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


def should_ingest_changed_file(fr: FileRef, cfg: ProjectConfig) -> bool:
    """Decide if the fileRef sould be processed further.

    Checks for duplicate in database (already ingested) by mtime & byte size.
    """
    engine = create_engine_from_url(cfg.storage.database_url)
    conn = engine.connect()
    sel = select(documents.c.doc_id, documents.c.byte_size, documents.c.mtime_ns).where(
        documents.c.doc_id == fr.relative_path)
    res = conn.execute(sel).fetchone()
    conn.close()

    if not res:
        # new file
        return True
    if res.byte_size == fr.file_size and res.mtime_ns == fr.mtime:
        # unchanged file
        return False
    return True
