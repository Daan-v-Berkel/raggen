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
from raggen.core.parsing.MarkdownParser import MarkdownParser
from raggen.core.chunking.chunker import ChunkerRegistry
from raggen.core.embeddings.embedder import (
    LocalSentenceTransformerEmbedder,
    embed_chunks,
)
from raggen.core.store import store_document_bundle, delete_documents, load_vector_backend, resolve_vector_backend_import
from raggen.core.store.metadata_store import fetch_all_document_ids
from raggen.core.scanner import scan_files
from raggen.core.runtime import get_engine
from raggen.core.results.envelope import ResultEnvelope, ResultMessage, init_result
from datetime import datetime
import json
from raggen.core.runs.store import get_run_store
from raggen.core.runs.decorators import persist_result


@persist_result(get_run_store)
def do_ingest(destructive: bool = False) -> ResultEnvelope:
    cfg = ProjectConfig.get_config()
    engine = get_engine()
    backend = load_vector_backend(
        resolve_vector_backend_import(cfg.storage.backend_key, cfg.storage.vector_backend_import)
    )

    ingest_result = init_result("ingest")

    # total files scanned includes skipped empty files
    registry = ParserRegistry(fallback_parser=PlainTextFallbackParser())
    registry.register(MarkdownParser())
    parser_service = ParserService(registry)
    embedder = LocalSentenceTransformerEmbedder(
        cfg.embedding.model_id,
        cache_dir=cfg.embedding.model_cache_dir,
        batch_size=cfg.embedding.batch_size,
        normalize=cfg.embedding.normalize,
    )
    doc_count = 0
    chunk_count = 0
    emb_count = 0
    skip_count = 0

    current_files = set()
    db_files = set(fetch_all_document_ids(engine))

    scanned = scan_files(
        cfg.project_root,
        ignore_filenames=cfg.scan.ignore_files,
        ignore_patterns=cfg.scan.ignore,
    )

    chunk_registry = ChunkerRegistry()

    for group, file_refs in scanned.groups.items():
        if not file_refs:
            continue

        chunker = chunk_registry.get(group)

        for fr in file_refs:
            current_files.add(fr.relative_path)
            # gating: raw bytes
            if not should_ingest_changed_file(fr, cfg):
                m = f"Skipping {
                    fr.relative_path}: file already ingested and unchanged"
                logger.warning(m)
                ingest_result.warnings.append(ResultMessage(
                    code="unchanged", message=m))
                skip_count += 1
                continue
            try:
                data = Path(fr.path).read_bytes()
            except Exception:
                m = f"Skipping {fr.relative_path}: could not read file"
                logger.warning(m)
                ingest_result.warnings.append(ResultMessage(
                    code="read_error", message=m))
                skip_count += 1
                continue
            ok, reason = should_ingest_raw_bytes(data)
            if not ok:
                m = f"Skipping {fr.relative_path}: empty file (0 bytes)"
                logger.warning(m)
                ingest_result.warnings.append(ResultMessage(
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
                if result.encoding_error_ratio > cfg.scan.max_encoding_error_ratio:
                    m = (
                        f"Skipping {fr.relative_path}: "
                        f"{result.encoding_error_ratio:.1%} of content is invalid UTF-8 "
                        f"(threshold: {cfg.scan.max_encoding_error_ratio:.1%}). "
                        "File is likely binary or severely corrupted."
                    )
                    logger.warning(m)
                    ingest_result.warnings.append(
                        ResultMessage(code="binary_or_corrupt", message=m)
                    )
                    skip_count += 1
                    continue
                for w in result.warnings:
                    logger.warning(w)
                    ingest_result.warnings.append(
                        ResultMessage(code="encoding_warning", message=w)
                    )
                doc = result.document
                # gating: parsed document
                ok2, reason2 = should_ingest_parsed_document(doc)
                if not ok2:
                    m = f"Skipping {fr.relative_path}: empty file"
                    logger.warning(m)
                    ingest_result.warnings.append(ResultMessage(
                        code="empty_file", message=m))
                    skip_count += 1
                    continue
                chunks = chunker.chunk(doc)
                # embed
                em_results = embed_chunks(embedder, chunks)
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
                ts = datetime.now().isoformat(timespec='seconds')
                for ch, em in zip(chunks, em_results):
                    chunk_rows.append(
                        {
                            "chunk_id": ch.chunk_id,
                            "doc_id": ch.doc_id,
                            "chunk_index": ch.chunk_index,
                            "text": ch.text,
                            "start_offset": ch.start_char if ch.start_char is not None else 0,
                            "end_offset": ch.end_char if ch.end_char is not None else len(ch.text),
                            "page_number": ch.metadata.page_start,
                            "heading_path_json": json.dumps(ch.metadata.section_path) if ch.metadata.section_path else None,
                            "chunk_config_hash": ch.config_hash,
                            "created_at": ts,
                        }
                    )
                    embedding_meta_rows.append(
                        {
                            "chunk_id": ch.chunk_id,
                            "embedding_model_id": cfg.embedding.model_id,
                            "dim": cfg.embedding.dim,
                            "normalized": cfg.embedding.normalize,
                            "created_at": ts,
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
                ingest_result.errors.append(ResultMessage(
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

    ingest_result.success = True
    ingest_result.data = {
        "summary": result_data,
        "details": result_data,
    }

    return ingest_result
