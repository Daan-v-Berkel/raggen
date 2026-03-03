from typing import Iterable, Any, Dict, List, Optional
from pathlib import Path
from .chunker import Chunker
from .chunks import ChunkConfig, Chunk
from ..parsing.parser import ParseInput, ParserService, ParserRegistry, SourceRef
from ..logger import get_logger

logger = get_logger("rag-engine.chunking.pipeline")


def _get_field(obj: Any, *names: str) -> Optional[Any]:
    """Try to retrieve a field from an object or dict using multiple candidate names."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        for n in names:
            if n in obj:
                return obj[n]
        return None
    for n in names:
        if hasattr(obj, n):
            val = getattr(obj, n)
            if val is not None:
                return val
    return None


def chunk_files(file_refs: Iterable[Any], conf: ChunkConfig, registry: ParserRegistry) -> Dict[str, List[Chunk]]:
    """Chunk a sequence of scanner fileRef-like objects.

    Parameters
    - file_refs: iterable of objects or dicts that provide at least `doc_id` and `data`.
      Supported field names (in priority order):
        doc_id: ("doc_id", "id", "name")
        data: ("data", "bytes", "raw")  -- bytes expected
        mimetype: ("mimetype", "mime", "content_type")
        filename: ("filename", "path", "name")

    - conf: ChunkConfig to use for chunking
    - registry: ParserRegistry used to resolve parsers

    Returns
    - mapping of document id -> list[Chunk]

    Behavior
    - Skips items without data and logs errors
    - Ensures Document.source is a SourceRef (or creates one from filename/doc_id)
    - Catches parse and chunking errors per-file and continues processing others
    """

    parser_service = ParserService(registry)
    out: Dict[str, List[Chunk]] = {}

    for idx, fr in enumerate(file_refs):
        # Resolve common fields from either dict or object
        doc_id = _get_field(fr, "doc_id", "id", "name") or f"unnamed-{idx}"
        data = _get_field(fr, "data", "bytes", "raw")
        mimetype = _get_field(fr, "mimetype", "mime", "content_type") or "application/octet-stream"
        filename = _get_field(fr, "filename", "path", "name")

        if data is None:
            logger.error("Skipping %s: no data provided", doc_id)
            continue

        # Ensure bytes
        if isinstance(data, str):
            data = data.encode("utf-8")

        try:
            inp = ParseInput(doc_id=doc_id, data=data, mimetype=mimetype, filename=filename)
        except Exception as e:
            logger.exception("Failed to construct ParseInput for %s: %s", doc_id, e)
            continue

        try:
            result = parser_service.parse_document(inp)
        except Exception as e:
            logger.exception("Parsing failed for %s: %s", doc_id, e)
            continue

        doc = result.document

        # Ensure doc.source is a SourceRef; create one if parser didn't set it
        if getattr(doc, "source", None) is None:
            try:
                rel = filename or doc.doc_id
                doc.source = SourceRef(scheme="file", rel_path=rel, display_name=Path(rel).name)
            except Exception:
                # ignore if assignment not allowed
                pass

        chunker = Chunker(doc)
        try:
            chunks = chunker.chunk(conf)
        except Exception as e:
            logger.exception("Chunking failed for %s: %s", doc.doc_id, e)
            continue

        # Ensure each chunk has source metadata populated
        for c in chunks:
            try:
                # c.metadata is a pydantic model; assign attribute when possible
                if getattr(c.metadata, "source", None) is None:
                    c.metadata.source = getattr(doc, "source", None)
            except Exception:
                # If metadata is a dict-like structure, attempt to set key
                try:
                    if isinstance(c.metadata, dict):
                        c.metadata["source"] = getattr(doc, "source", None)
                except Exception:
                    pass

        out[doc.doc_id] = chunks

    return out
