from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
from raggen.core.config.project import ProjectConfig
from raggen.core.ingest.logging import log_stage, log_error, logger
from raggen.core.ingest.gating import should_ingest_raw_bytes, should_ingest_parsed_document, should_ingest_changed_file
from raggen.core.parsing.parser import ParserRegistry, ParseInput, ParserService
from raggen.core.parsing.PlainTextParser import PlainTextFallbackParser
from raggen.core.chunking.chunks import DEFAULT_CHUNK_CONFIG
from raggen.core.chunking.chunker import Chunker
from raggen.core.embeddings.embedder import LocalSentenceTransformerEmbedder, embed_chunks
from raggen.core.store import init_database, store_document_bundle, delete_documents
from raggen.core.store.metadata_store import fetch_all_document_ids
from raggen.core.scanner import scan_files
from datetime import datetime


def do_ingest(destructive: bool = False) -> Dict[str, Any]:
    # init db
    cfg = ProjectConfig.get_config()
    engine = init_database(cfg, destructive=destructive)
    backend = getattr(engine, '_rag_vector_backend', None)
    # scan using scanner
    initial_warnings = {"empty_bytes": 0}
    # total files scanned includes skipped empty files
    registry = ParserRegistry(fallback_parser=PlainTextFallbackParser())
    parser_service = ParserService(registry)
    embedder = LocalSentenceTransformerEmbedder(cfg.embedding.model_id)
    doc_count = 0
    chunk_count = 0
    emb_count = 0
    errors = []
    # aggregate warnings
    warnings_agg: dict[str, int] = {}
    for k, v in initial_warnings.items():
        warnings_agg[k] = warnings_agg.get(k, 0) + v

    current_files = set()
    db_files = set(fetch_all_document_ids(engine))
    # TODO: add proper ignorefile and other config
    for fr in scan_files(cfg.project_root):
        current_files.add(fr.relative_path)
        # gating: raw bytes
        if not should_ingest_changed_file(fr, cfg):
            logger.warning("Skipping %s: file already ingested and unchanged",
                           fr.relative_path)
            warnings_agg['unchanged'] = warnings_agg.get('unchanged', 0) + 1
            continue
        try:
            data = Path(fr.path).read_bytes()
        except Exception:
            logger.warning("Skipping %s: could not read file",
                           fr.relative_path)
            warnings_agg['read_error'] = warnings_agg.get('read_error', 0) + 1
            continue
        ok, reason = should_ingest_raw_bytes(data)
        if not ok:
            logger.warning("Skipping %s: empty file (0 bytes)",
                           fr.relative_path)
            warnings_agg['empty_bytes'] = warnings_agg.get(
                'empty_bytes', 0) + 1
            continue

        try:
            doc_id = fr.relative_path
            mimetype = fr.mime_type or 'application/octet-stream'
            inp = ParseInput(doc_id=doc_id, data=data,
                             mimetype=mimetype, filename=Path(fr.path).name)
            result = parser_service.parse_document(inp)
            doc = result.document
            # gating: parsed document
            ok2, reason2 = should_ingest_parsed_document(doc)
            if not ok2:
                logger.warning(
                    "Skipping %s: parser produced empty text", doc_id)
                warnings_agg['empty_text_after_parse'] = warnings_agg.get(
                    'empty_text_after_parse', 0) + 1
                continue
            chunker = Chunker(doc)
            chunks = chunker.chunk(DEFAULT_CHUNK_CONFIG)
            # embed
            em_results = embed_chunks(
                embedder, chunks, batch_size=cfg.embedding.batch_size, normalize=cfg.embedding.normalize)
            # build rows
            document_row = {
                'doc_id': doc.doc_id,
                'source_path': str(doc.source or doc.doc_id),
                'mimetype': mimetype,
                'mtime_ns': fr.mtime,
                'byte_size': fr.file_size,
                'content_hash': fr.content_hash,
                'parsed_at': datetime.now(),
                'parser_id': 'plain',
                'structure_version': 'v1',
                'text_char_len': len(doc.text),
            }
            chunk_rows = []
            embedding_meta_rows = []
            vectors = []
            for ch, em in zip(chunks, em_results):
                chunk_rows.append({
                    'chunk_id': ch.chunk_id,
                    'doc_id': ch.doc_id,
                    'chunk_index': ch.chunk_index,
                    'text': ch.text,
                    'start_offset': getattr(ch, 'start_offset', 0) or 0,
                    'end_offset': getattr(ch, 'end_offset', len(ch.text)) or len(ch.text),
                    'page_number': getattr(ch, 'page_number', None),
                    'heading_path_json': getattr(ch, 'heading_path_json', None),
                    'chunk_config_hash': cfg.chunking.chunk_size or '',
                    'created_at': '',
                })
                embedding_meta_rows.append({
                    'chunk_id': ch.chunk_id,
                    'embedding_model_id': cfg.embedding.model_id,
                    'dim': cfg.embedding.dim,
                    'normalized': 1 if cfg.embedding.normalize else 0,
                    'created_at': '',
                })
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
            log_error(str(fr.path), 'ingest', exc)
            errors.append({'path': str(fr.path), 'error': str(exc)})
    log_stage('ingest_done', {'docs': doc_count,
              'chunks': chunk_count, 'embeddings': emb_count})
    # compute skipped/docs parsed/docs deleted
    docs_skipped = sum(warnings_agg.values())
    docs_to_remove = db_files - current_files
    n_removed = delete_documents(engine=engine,
                                 vector_backend=backend, documents=docs_to_remove)
    result = {
        'docs_parsed': doc_count,
        'docs_skipped': docs_skipped,
        'skip_reasons': warnings_agg,
        'docs_removed': n_removed,
        'docs': doc_count,
        'chunks': chunk_count,
        'embeddings': emb_count,
        'errors': errors,
    }
    return result
