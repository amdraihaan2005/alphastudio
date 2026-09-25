from uuid import UUID
from sqlalchemy.orm import Session, joinedload
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
    Eager-loads the parent SourceDocument to eliminate N+1 relationship queries.
    Scopes results to public docs (user_id IS NULL) plus the caller's private uploads.
    """
    query = (
        db.query(DocumentChunk)
        .options(joinedload(DocumentChunk.document))
        .join(SourceDocument, DocumentChunk.source_document_id == SourceDocument.id)
    )
    if user_id:
        query = query.filter(
            or_(SourceDocument.user_id.is_(None), SourceDocument.user_id == user_id)
        )
    else:
        query = query.filter(SourceDocument.user_id.is_(None))

    return (
        query.order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
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
    Executes a PostgreSQL full-text search matching against search_vector.
    Uses websearch_to_tsquery for natural multi-word questions (Google-style syntax),
    falling back to plainto_tsquery if websearch returns no matches.
    Eager-loads the parent SourceDocument to eliminate N+1 relationship queries.
    """
    cleaned_query = query_text.strip()
    if not cleaned_query:
        return []

    # websearch_to_tsquery handles natural language questions gracefully
    tsquery = func.websearch_to_tsquery(TEXT_SEARCH_CONFIG, cleaned_query)

    query = (
        db.query(DocumentChunk)
        .options(joinedload(DocumentChunk.document))
        .join(SourceDocument, DocumentChunk.source_document_id == SourceDocument.id)
        .filter(DocumentChunk.search_vector.op("@@")(tsquery))
    )
    if user_id:
        query = query.filter(
            or_(SourceDocument.user_id.is_(None), SourceDocument.user_id == user_id)
        )
    else:
        query = query.filter(SourceDocument.user_id.is_(None))

    results = (
        query.order_by(func.ts_rank(DocumentChunk.search_vector, tsquery).desc())
        .limit(limit)
        .all()
    )

    # Fallback if websearch returns empty on highly specific punctuation/phrasing
    if not results:
        plain_query = func.plainto_tsquery(TEXT_SEARCH_CONFIG, cleaned_query)
        fallback_query = (
            db.query(DocumentChunk)
            .options(joinedload(DocumentChunk.document))
            .join(SourceDocument, DocumentChunk.source_document_id == SourceDocument.id)
            .filter(DocumentChunk.search_vector.op("@@")(plain_query))
        )
        if user_id:
            fallback_query = fallback_query.filter(
                or_(SourceDocument.user_id.is_(None), SourceDocument.user_id == user_id)
            )
        else:
            fallback_query = fallback_query.filter(SourceDocument.user_id.is_(None))

        results = (
            fallback_query.order_by(
                func.ts_rank(DocumentChunk.search_vector, plain_query).desc()
            )
            .limit(limit)
            .all()
        )

    return results
