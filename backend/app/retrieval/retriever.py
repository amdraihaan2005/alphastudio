from uuid import UUID
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_, and_
from app.ingest.embedder import get_query_embedding
from app.retrieval.queries import semantic_search, full_text_search
from app.retrieval.fusion import reciprocal_rank_fusion
from app.database.models.document_chunk import DocumentChunk


def retrieve_hybrid(
    db: Session,
    query_text: str,
    user_id: UUID | None = None,
    limit: int = 5,
    fetch_neighbors: bool = True,
) -> list[dict]:
    """
    Main retrieval entry point.
    Given a query string, it generates the search query embedding via Cohere,
    runs semantic and keyword full-text searches (scoped to the user_id),
    applies RRF to merge the results, fetches adjacent context chunks in a single batched query,
    and formats the output dictionary for RAG use.

    user_id = None  → public filings only.
    user_id = <uuid> → public filings + that user's private uploads.
    """
    if not query_text.strip():
        return []

    query_embedding = get_query_embedding(query_text)

    candidate_limit = limit * 4
    semantic_results = semantic_search(
        db, query_embedding, user_id=user_id, limit=candidate_limit
    )
    fts_results = full_text_search(
        db, query_text, user_id=user_id, limit=candidate_limit
    )

    fused_results = reciprocal_rank_fusion(semantic_results, fts_results, limit=limit)

    neighbor_map = {}
    if fetch_neighbors and fused_results:
        # Collect neighbor target conditions to query in a single batched SQL query
        neighbor_conditions = []
        for chunk, _ in fused_results:
            target_indices = []
            if chunk.chunk_index > 0:
                target_indices.append(chunk.chunk_index - 1)
            target_indices.append(chunk.chunk_index + 1)

            if target_indices:
                neighbor_conditions.append(
                    and_(
                        DocumentChunk.source_document_id == chunk.source_document_id,
                        DocumentChunk.chunk_index.in_(target_indices),
                    )
                )

        if neighbor_conditions:
            neighbor_chunks = (
                db.query(DocumentChunk)
                .options(joinedload(DocumentChunk.document))
                .filter(or_(*neighbor_conditions))
                .all()
            )
            for nc in neighbor_chunks:
                neighbor_map[(nc.source_document_id, nc.chunk_index)] = nc

    results = []
    for chunk, score in fused_results:
        preceding_chunk = neighbor_map.get(
            (chunk.source_document_id, chunk.chunk_index - 1)
        )
        succeeding_chunk = neighbor_map.get(
            (chunk.source_document_id, chunk.chunk_index + 1)
        )

        results.append(
            {
                "chunk": chunk,
                "score": score,
                "preceding_chunk": preceding_chunk,
                "succeeding_chunk": succeeding_chunk,
            }
        )

    return results
