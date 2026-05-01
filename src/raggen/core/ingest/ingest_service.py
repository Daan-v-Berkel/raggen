from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional
from raggen.core.config.project import ProjectConfig
from raggen.core.embeddings.model_specs_cache import ModelSpecsCache, MissingModelSpecsError
from raggen.core.ingest.gating import (
    should_ingest_raw_bytes,
    should_ingest_parsed_document,
    should_ingest_changed_file,
)
from raggen.core.parsing.parser import ParserRegistry, ParseInput, ParserService
from raggen.core.parsing.PlainTextParser import PlainTextFallbackParser
from raggen.core.parsing.MarkdownParser import MarkdownParser
from raggen.core.parsing.HtmlParser import HtmlParser
from raggen.core.chunking.chunker import ChunkerRegistry
from raggen.core.embeddings.embedder import (
    create_embedder,
    embed_chunks,
)
from raggen.core.store import store_document_bundle, delete_documents, load_vector_backend, resolve_vector_backend_import
from raggen.core.store.metadata_store import fetch_all_document_ids
from raggen.core.scanner import scan_files
from raggen.core.runtime import get_engine
from raggen.core.results.envelope import ResultEnvelope, ResultMessage, init_result
from raggen.core.metadata.store import load_project_state, create_project_state, save_project_state
from raggen.core.metadata.models import ProjectLifecycleState
from raggen.core.validation.project_validator import ProjectValidator
from datetime import datetime
import json
from raggen.core.runs.store import get_run_store
from raggen.core.runs.decorators import persist_result


@persist_result(get_run_store)
def do_ingest(
    force: bool = False,
    on_file: Optional[Callable[[], None]] = None,
) -> ResultEnvelope:
    cfg = ProjectConfig.get_config()

    # Resolve model capabilities from the cache written by `rag build`.
    # This fires before any model load so the error surfaces immediately.
    _specs_dir = cfg.project_root / ".rag" / "metadata" / "model_specs"
    _cache = ModelSpecsCache(_specs_dir)
    caps = _cache.get(cfg.embedding.model_id)
    if caps is None:
        raise MissingModelSpecsError(cfg.embedding.model_id)

    # Resolve dim: if not pinned in config, use the cached actual dim.
    # After this, cfg.embedding.dim is always a concrete integer for downstream code.
    cfg.embedding.dim = cfg.embedding.dim or caps.actual_dim

    engine = get_engine()
    backend = load_vector_backend(
        resolve_vector_backend_import(
            cfg.storage.backend_key, cfg.storage.vector_backend_import)
    )

    ingest_result = init_result("ingest")

    # ---------------------------------------------------------------------------
    # Unified pre-flight validation (Steps 3, 5, 6).
    # All errors are raised before any file is touched; advisory warnings are
    # returned and added to the result envelope.
    # ---------------------------------------------------------------------------
    _project_state = load_project_state(cfg.project_root)
    _validation_warnings = ProjectValidator.validate_for_ingest(
        cfg, engine, _project_state, caps
    )
    ingest_result.warnings.extend(_validation_warnings)

    # total files scanned includes skipped empty files
    registry = ParserRegistry(fallback_parser=PlainTextFallbackParser())
    registry.register(MarkdownParser())
    registry.register(HtmlParser())
    parser_service = ParserService(registry)
    embedder = create_embedder(
        model_id=cfg.embedding.model_id,
        backend=cfg.embedding.backend,
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
    # Build the token length function once — cheap after the model is loaded.
    # Used by any group configured with unit = "tokens".
    _token_length_fn = embedder.get_length_function()

    for group, file_refs in scanned.groups.items():
        if not file_refs:
            continue

        group_conf = cfg.chunking.get(group)
        length_fn = _token_length_fn if (
            group_conf and group_conf.unit == "tokens") else len
        chunker = chunk_registry.get(group, length_function=length_fn)

        for fr in file_refs:
            if on_file:
                on_file()
            current_files.add(fr.relative_path)
            # gating: raw bytes
            if not force:
                if not should_ingest_changed_file(fr, cfg):
                    skip_count += 1
                    continue
            try:
                data = Path(fr.path).read_bytes()
            except Exception:
                m = f"Skipping {fr.relative_path}: could not read file"
                ingest_result.warnings.append(ResultMessage(
                    code="read_error", message=m))
                skip_count += 1
                continue
            ok, reason = should_ingest_raw_bytes(data)
            if not ok:
                # Empty files are counted but not individually warned about —
                # they're a normal edge-case covered by the skipped total.
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
                        f"(threshold: {
                            cfg.scan.max_encoding_error_ratio:.1%}). "
                        "File is likely binary or severely corrupted."
                    )
                    ingest_result.warnings.append(
                        ResultMessage(code="binary_or_corrupt", message=m)
                    )
                    skip_count += 1
                    continue
                for w in result.warnings:
                    ingest_result.warnings.append(
                        ResultMessage(code="encoding_warning", message=w)
                    )
                doc = result.document
                # gating: parsed document
                ok2, reason2 = should_ingest_parsed_document(doc)
                if not ok2:
                    # Parsed-but-empty files are counted but not individually
                    # warned about — covered by the skipped total.
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
                ingest_result.errors.append(ResultMessage(
                    code="ingestion_error", message=f"path: {str(fr.path)}, error: {str(exc)}"))

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

    # Forced re-ingest with no errors means all files reflect the current config.
    # Update the project snapshot so chunking drift warnings clear on next run.
    if force and not ingest_result.errors:
        new_state = create_project_state(cfg=cfg, state=ProjectLifecycleState.SET_UP)
        save_project_state(new_state)

    return ingest_result
