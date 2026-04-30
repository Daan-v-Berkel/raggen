# Lazy re-export: importing this package does not pull in ingest_service
# (and transitively sentence_transformers / torch) until do_ingest is
# actually accessed.  This keeps `rag ingest` startup near-instant.

__all__ = ["do_ingest"]


def __getattr__(name: str):
    if name == "do_ingest":
        from .ingest_service import do_ingest
        return do_ingest
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
