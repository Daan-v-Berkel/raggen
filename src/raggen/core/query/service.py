from __future__ import annotations

from raggen.core.query.models import QueryRequest, QueryResponse, RetrievedChunk
from raggen.core.store.metadata_store import MetadataStore
from raggen.core.runtime import get_engine
from raggen.core.config.project import ProjectConfig
from raggen.core.embeddings.embedder import LocalSentenceTransformerEmbedder
from raggen.core.store.plugin_loader import load_vector_backend


def query(request: QueryRequest) -> QueryResponse:
    """
    Retrieval-only query flow.

    Steps:
      1. load config + engine
      2. validate query embedding configuration
      3. embed query text
      4. call vector backend search(...)
      5. fetch chunk rows from metadata store
      6. build QueryResponse

    Returns structured data only. Does not print.
    """
    cfg = ProjectConfig.get_config()
    engine = get_engine()

    # if engine is None:
    #     raise RuntimeError(
    #         "No database engine available. Did you run bootstrap()?")

    vector_backend = load_vector_backend(cfg.storage.vector_backend_import)

    query_model_id = _resolve_query_model_id(request, cfg)
    query_dim = _resolve_query_dim(cfg)
    normalize = bool(cfg.embedding.normalize)

    embedder = LocalSentenceTransformerEmbedder(model_id=query_model_id)
    _validate_query_embedder(embedder, expected_dim=query_dim)

    matrix = embedder.embed_texts(
        [request.text],
        batch_size=1,
        normalize=normalize,
    )
    query_vector = matrix[0].tolist()

    search_results = vector_backend.search(
        engine,
        query_vector=query_vector,
        top_k=request.top_k,
    )

    if not search_results:
        return QueryResponse(
            query=request.text,
            matches=[],
            answer=None,
            used_query_model=query_model_id,
            used_llm_model=request.llm_model_id or None,
        )

    chunk_ids = [chunk_id for chunk_id, _score in search_results]
    score_by_chunk_id = {chunk_id: score for chunk_id, score in search_results}

    store = MetadataStore(engine)
    with engine.connect() as conn:
        rows = store.fetch_chunks_by_ids(conn, chunk_ids)

    matches: list[RetrievedChunk] = []
    for row in rows:
        chunk_id = row["chunk_id"]
        matches.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=row["doc_id"],
                text=row["text"],
                score=float(score_by_chunk_id[chunk_id]),
                chunk_index=int(row["chunk_index"]),
                start_offset=row.get("start_offset"),
                end_offset=row.get("end_offset"),
                page_number=row.get("page_number"),
                heading_path=row.get("heading_path"),
            )
        )

    return QueryResponse(
        query=request.text,
        matches=matches,
        answer=None,
        used_query_model=query_model_id,
        used_llm_model=request.llm_model_id or None,
    )


def _resolve_query_model_id(request: QueryRequest, cfg: ProjectConfig) -> str:
    """
    Query model precedence:
      1. request override
      2. cfg.query.model_id if set
      3. fallback to ingestion embedding model
    """
    if request.query_model_id:
        return request.query_model_id

    if cfg.query.model_id:
        return cfg.query.model_id

    return cfg.embedding.model_id


def _resolve_query_dim(cfg: ProjectConfig) -> int:
    """
    For now, query embedding dim must match stored embedding dim.
    """
    return int(cfg.embedding.dim)


def _validate_query_embedder(embedder, *, expected_dim: int) -> None:
    actual_dim = int(embedder.dim)
    if actual_dim != expected_dim:
        raise ValueError(
            f"Query embedding dimension mismatch: model returns {actual_dim}, "
            f"but project expects {expected_dim}."
        )
