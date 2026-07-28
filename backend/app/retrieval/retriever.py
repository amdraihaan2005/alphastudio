from uuid import UUID
from sqlalchemy.orm import Session
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
    applies RRF to merge the results, fetches adjacent context chunks,
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

    results = []

    for chunk, score in fused_results:
        preceding_chunk = None
        succeeding_chunk = None

        if fetch_neighbors:
            preceding_chunk = (
                db.query(DocumentChunk)
                .filter(
                    DocumentChunk.source_document_id == chunk.source_document_id,
                    DocumentChunk.chunk_index == chunk.chunk_index - 1,
                )
                .first()
            )

            succeeding_chunk = (
                db.query(DocumentChunk)
                .filter(
                    DocumentChunk.source_document_id == chunk.source_document_id,
                    DocumentChunk.chunk_index == chunk.chunk_index + 1,
                )
                .first()
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
