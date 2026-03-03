from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
from raggen.core.ingest.config import ProjectConfig, default_project_config
from raggen.core.ingest.logging import log_stage, log_error
from raggen.core.parsing.parser import ParserRegistry, ParseInput, ParserService
from raggen.core.parsing.PlainTextParser import PlainTextFallbackParser
from raggen.core.chunking.chunks import DEFAULT_CHUNK_CONFIG
from raggen.core.chunking.chunker import Chunker
from raggen.core.embeddings.embedder import LocalSentenceTransformerEmbedder, embed_chunks
from raggen.core.store import RagInitConfig, init_database, load_vector_backend, store_document_bundle
import os
import fnmatch


def _load_files(root: Path, ignore_patterns: list[str]):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d not in {".git", "node_modules", "__pycache__"}]
        for fn in filenames:
            if fn.startswith('.'):
                continue
            p = Path(dirpath) / fn
            rel = str(p.relative_to(root)).replace(os.sep, '/')
            skip = False
            for pat in ignore_patterns:
                if pat.endswith('/') and rel.startswith(pat.rstrip('/')):
                    skip = True
                    break
                if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(fn, pat):
                    skip = True
                    break
            if not skip:
                files.append(p)
    return files


def _build_rag_init_config(cfg: ProjectConfig) -> RagInitConfig:
    notes = {"vector_backend_import": cfg.storage.vector_backend_import}
    return RagInitConfig(
        backend_key=cfg.storage.backend_key,
        vector_backend_import=cfg.storage.vector_backend_import,
        database_url=cfg.storage.database_url,
        embedding_model_id=cfg.embedding.model_id,
        embedding_dim=cfg.embedding.dim,
        embedding_normalized=cfg.embedding.normalize,
        chunk_config_hash=DEFAULT_CHUNK_CONFIG.get('chunk_config_hash', ''),
        notes=notes,
    )


def init_and_ingest(*, cfg: ProjectConfig, destructive_init: bool = False) -> Dict[str, Any]:
    root = Path(cfg.project_root)
    root.mkdir(parents=True, exist_ok=True)
    os.makedirs(root / '.rag', exist_ok=True)
    # init db
    rag_cfg = _build_rag_init_config(cfg)
    engine = init_database(rag_cfg, destructive=destructive_init)
    backend = getattr(engine, '_rag_vector_backend', None)
    # scan
    files = _load_files(root, cfg.scan.ignore if cfg.scan.ignore else [])
    log_stage('scan', {'files': len(files)})
    registry = ParserRegistry(fallback_parser=PlainTextFallbackParser())
    parser_service = ParserService(registry)
    embedder = LocalSentenceTransformerEmbedder(cfg.embedding.model_id)
    doc_count = 0
    chunk_count = 0
    emb_count = 0
    errors = []
    for p in files:
        try:
            data = p.read_bytes()
            doc_id = str(p.relative_to(root))
            mimetype = 'application/octet-stream'
            inp = ParseInput(doc_id=doc_id, data=data, mimetype=mimetype, filename=p.name)
            result = parser_service.parse_document(inp)
            doc = result.document
            chunker = Chunker(doc)
            chunks = chunker.chunk(DEFAULT_CHUNK_CONFIG)
            # embed
            em_results = embed_chunks(embedder, chunks, batch_size=cfg.embedding.batch_size, normalize=cfg.embedding.normalize)
            # build rows
            document_row = {
                'doc_id': doc.doc_id,
                'source_path': str(doc.source or doc.doc_id),
                'mimetype': mimetype,
                'byte_size': len(data),
                'content_hash': '',
                'parsed_at': '',
                'parser_id': 'plain',
                'structure_version': 'v1',
                'text_char_len': len(doc.canonical_text),
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
                cfg=rag_cfg,
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
            log_error(str(p), 'ingest', exc)
            errors.append({'path': str(p), 'error': str(exc)})
    log_stage('ingest_done', {'docs': doc_count, 'chunks': chunk_count, 'embeddings': emb_count})
    return {'docs': doc_count, 'chunks': chunk_count, 'embeddings': emb_count, 'errors': errors}


def ingest_only(*, cfg: ProjectConfig) -> Dict[str, Any]:
    return init_and_ingest(cfg=cfg, destructive_init=False)
