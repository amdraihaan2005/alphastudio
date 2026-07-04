from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import func, or_
from app.database.models.document_chunk import DocumentChunk
from app.database.models.source_document import SourceDocument
from app.database.models.constants import TEXT_SEARCH_CONFIG


def semantic_search(
    db: Session,
    query_embedding: list[float],
    user_id: UUID | None = None,
    limit: int = 20,
) -> list[DocumentChunk]:
    """
    Executes a pgvector semantic similarity search using cosine distance.
    Scopes results to public docs (user_id IS NULL) plus the caller's private uploads.
    Returns the top matching DocumentChunks ordered by similarity.
    """
    query = (
        db.query(DocumentChunk)
        .join(SourceDocument, DocumentChunk.source_document_id == SourceDocument.id)
    )
    if user_id:
        query = query.filter(
            or_(SourceDocument.user_id.is_(None), SourceDocument.user_id == user_id)
        )
    else:
        query = query.filter(SourceDocument.user_id.is_(None))

    return (
        query
        .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
        .limit(limit)
        .all()
    )


def full_text_search(
    db: Session,
    query_text: str,
    user_id: UUID | None = None,
    limit: int = 20,
) -> list[DocumentChunk]:
    """
    Executes a PostgreSQL plain full-text search matching against search_vector.
    Scopes results to public docs (user_id IS NULL) plus the caller's private uploads.
    Returns the top matching DocumentChunks ordered by FTS ts_rank descending.
    """
    if not query_text.strip():
        return []

    tsquery = func.plainto_tsquery(TEXT_SEARCH_CONFIG, query_text)

    query = (
        db.query(DocumentChunk)
        .join(SourceDocument, DocumentChunk.source_document_id == SourceDocument.id)
        .filter(DocumentChunk.search_vector.op("@@")(tsquery))
    )
    if user_id:
        query = query.filter(
            or_(SourceDocument.user_id.is_(None), SourceDocument.user_id == user_id)
        )
    else:
        query = query.filter(SourceDocument.user_id.is_(None))

    return (
        query
        .order_by(func.ts_rank(DocumentChunk.search_vector, tsquery).desc())
        .limit(limit)
        .all()
    )
