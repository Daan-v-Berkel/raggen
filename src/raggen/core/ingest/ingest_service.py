from __future__ import annotations

from pathlib import Path
from raggen.core.config.project import ProjectConfig
from raggen.core.ingest.logging import log_stage, log_error, logger
from raggen.core.ingest.gating import (
    should_ingest_raw_bytes,
    should_ingest_parsed_document,
    should_ingest_changed_file,
)
from raggen.core.parsing.parser import ParserRegistry, ParseInput, ParserService
from raggen.core.parsing.PlainTextParser import PlainTextFallbackParser
from raggen.core.chunking.chunker import ChunkerRegistry
from raggen.core.embeddings.embedder import (
    LocalSentenceTransformerEmbedder,
    embed_chunks,
)
from raggen.core.store import store_document_bundle, delete_documents, load_vector_backend
from raggen.core.store.metadata_store import fetch_all_document_ids
from raggen.core.scanner import scan_files
from raggen.core.runtime import get_engine
from raggen.core.results.envelope import ResultEnvelope, ResultMessage
from datetime import datetime


def do_ingest(destructive: bool = False) -> ResultEnvelope:
    cfg = ProjectConfig.get_config()
    engine = get_engine()
    backend = load_vector_backend(cfg.storage.vector_backend_import)

    ingest_result = ResultEnvelope(
        operation="ingest",
        success=False,
    )
    warnings_array: list[ResultMessage] = []
    errors_array: list[ResultMessage] = []

    # total files scanned includes skipped empty files
    registry = ParserRegistry(fallback_parser=PlainTextFallbackParser())
    parser_service = ParserService(registry)
    embedder = LocalSentenceTransformerEmbedder(cfg.embedding.model_id)
    doc_count = 0
    chunk_count = 0
    emb_count = 0
    skip_count = 0

    current_files = set()
    db_files = set(fetch_all_document_ids(engine))

    scanned = scan_files(
        cfg.project_root, ignore_filenames=cfg.scan.ignore_files)

    chunk_registry = ChunkerRegistry()

    for group, file_refs in scanned.groups.items():
        if not file_refs:
            continue

        chunker = chunk_registry.get(group)

        for fr in file_refs:
            current_files.add(fr.relative_path)
            # gating: raw bytes
            if not should_ingest_changed_file(fr, cfg):
                m = f"Skipping {fr.relative_path}: file already ingested and unchanged"
                logger.warning(m)
                warnings_array.append(ResultMessage(
                    code="unchanged", message=m))
                skip_count += 1
                continue
            try:
                data = Path(fr.path).read_bytes()
            except Exception:
                m = f"Skipping {fr.relative_path}: could not read file"
                logger.warning(m)
                warnings_array.append(ResultMessage(
                    code="read_error", message=m))
                skip_count += 1
                continue
            ok, reason = should_ingest_raw_bytes(data)
            if not ok:
                m = f"Skipping {fr.relative_path}: empty file (0 bytes)"
                logger.warning(m)
                warnings_array.append(ResultMessage(
                    code="zero_bytes", message=m))
                skip_count += 1
                continue

            try:
                doc_id = fr.relative_path
                mimetype = fr.mime_type or "application/octet-stream"
                inp = ParseInput(
                    doc_id=doc_id,
                    data=data,
                    mimetype=mimetype,
                    filename=Path(fr.path).name,
                )
                result = parser_service.parse_document(inp)
                doc = result.document
                # gating: parsed document
                ok2, reason2 = should_ingest_parsed_document(doc)
                if not ok2:
                    m = f"Skipping {fr.relative_path}: empty file"
                    logger.warning(m)
                    warnings_array.append(ResultMessage(
                        code="empty_file", message=m))
                    skip_count += 1
                    continue
                chunks = chunker.chunk(doc)
                # embed
                em_results = embed_chunks(
                    embedder,
                    chunks,
                    batch_size=cfg.embedding.batch_size,
                    normalize=cfg.embedding.normalize,
                )
                # build rows
                document_row = {
                    "doc_id": doc.doc_id,
                    "source_path": str(doc.source or doc.doc_id),
                    "mimetype": mimetype,
                    "mtime_ns": fr.mtime,
                    "byte_size": fr.file_size,
                    "content_hash": fr.content_hash,
                    "parsed_at": datetime.now(),
                    "parser_id": "plain",
                    "structure_version": "v1",
                    "text_char_len": len(doc.text),
                }
                chunk_rows = []
                embedding_meta_rows = []
                vectors = []
                for ch, em in zip(chunks, em_results):
                    chunk_rows.append(
                        {
                            "chunk_id": ch.chunk_id,
                            "doc_id": ch.doc_id,
                            "chunk_index": ch.chunk_index,
                            "text": ch.text,
                            "start_offset": getattr(ch, "start_offset", 0) or 0,
                            "end_offset": getattr(ch, "end_offset", len(ch.text))
                            or len(ch.text),
                            "page_number": getattr(ch, "page_number", None),
                            "heading_path_json": getattr(ch, "heading_path_json", None),
                            "chunk_config_hash": ch.config_hash,
                            "created_at": "",
                        }
                    )
                    embedding_meta_rows.append(
                        {
                            "chunk_id": ch.chunk_id,
                            "embedding_model_id": cfg.embedding.model_id,
                            "dim": cfg.embedding.dim,
                            "normalized": 1 if cfg.embedding.normalize else 0,
                            "created_at": "",
                        }
                    )
                    vectors.append((ch.chunk_id, em.vector.tolist()))
                # store
                store_document_bundle(
                    engine=engine,
                    cfg=cfg,
                    vector_backend=backend,
                    document_row=document_row,
                    chunk_rows=chunk_rows,
                    embeddings=vectors,
                    embedding_meta_rows=embedding_meta_rows,
                )
                doc_count += 1
                chunk_count += len(chunk_rows)
                emb_count += len(vectors)
            except Exception as exc:
                log_error(str(fr.path), "ingest", exc)
                errors_array.append(ResultMessage(
                    code="ingestion_error", message=f"path: {str(fr.path)}, error: {str(exc)}"))

    log_stage(
        "ingest_done",
        {"docs": doc_count, "chunks": chunk_count, "embeddings": emb_count},
    )
    # compute skipped/docs parsed/docs deleted
    docs_to_remove = db_files - current_files
    n_removed = delete_documents(
        engine=engine, vector_backend=backend, documents=docs_to_remove
    )
    result_data = {
        "docs_parsed": doc_count,
        "docs_skipped": skip_count,
        "docs_removed": n_removed,
        "chunks": chunk_count,
        "embeddings": emb_count,
    }

    ingest_result.warnings = [
        ResultMessage(code=w.code, message=w.message)
        for w in warnings_array
    ]
    ingest_result.errors = [
        ResultMessage(code=e.code, message=e.message)
        for e in errors_array
    ]
    ingest_result.success = True
    ingest_result.data = result_data

    return ingest_result
