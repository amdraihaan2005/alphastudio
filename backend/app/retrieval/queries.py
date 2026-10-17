"""
Database Queries Module: Semantic & Full-Text Retrieval Engines.

Provides:
- `semantic_search`: Cosine distance similarity search via pgvector.
- `full_text_search`: Natural language text search via PostgreSQL websearch/plainto_tsquery.
- `semantic_search_documents`: Direct emission of canonical LangChain `Document` objects.
- `full_text_search_documents`: Direct emission of canonical LangChain `Document` objects.
- `document_chunk_to_langchain_document`: Provenance-preserving transformer for database chunks.
"""

from typing import Any, Sequence
from uuid import UUID

from langchain_core.documents import Document
from sqlalchemy import func, or_
from sqlalchemy.orm import Query, Session, joinedload

from app.database.models.constants import TEXT_SEARCH_CONFIG
from app.database.models.document_chunk import DocumentChunk
from app.database.models.source_document import SourceDocument


def _apply_document_filters(
    query: Query,
    user_id: UUID | None = None,
    document_ids: Sequence[UUID] | None = None,
    filenames: Sequence[str] | None = None,
) -> Query:
    """
    Applies consistent multi-tenant user isolation, document ID filtering,
    and filename scoping across all SQL query instances.
    """
    if user_id:
        query = query.filter(
            or_(SourceDocument.user_id.is_(None), SourceDocument.user_id == user_id)
        )
    else:
        query = query.filter(SourceDocument.user_id.is_(None))

    if document_ids:
        query = query.filter(SourceDocument.id.in_(document_ids))

    if filenames:
        query = query.filter(SourceDocument.filename.in_(filenames))

    return query


def semantic_search(
    db: Session,
    query_embedding: list[float],
    user_id: UUID | None = None,
    document_ids: Sequence[UUID] | None = None,
    filenames: Sequence[str] | None = None,
    limit: int = 20,
) -> list[DocumentChunk]:
    """
    Executes a pgvector semantic similarity search using cosine distance.
    Eager-loads the parent SourceDocument to eliminate N+1 relationship queries.
    Allows targeting specific document IDs or filenames for precise multi-document comparisons.
    """
    if not query_embedding:
        return []

    base_query = (
        db.query(DocumentChunk)
        .options(joinedload(DocumentChunk.document))
        .join(SourceDocument, DocumentChunk.source_document_id == SourceDocument.id)
    )
    filtered_query = _apply_document_filters(
        base_query, user_id=user_id, document_ids=document_ids, filenames=filenames
    )

    return (
        filtered_query.order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
        .limit(limit)
        .all()
    )


def full_text_search(
    db: Session,
    query_text: str,
    user_id: UUID | None = None,
    document_ids: Sequence[UUID] | None = None,
    filenames: Sequence[str] | None = None,
    limit: int = 20,
) -> list[DocumentChunk]:
    """
    Executes a PostgreSQL full-text search matching against search_vector.
    Uses websearch_to_tsquery for natural multi-word questions (Google-style syntax),
    falling back to plainto_tsquery if websearch returns no matches.
    Allows targeting specific document IDs or filenames for multi-file comparisons.
    """
    if not query_text or not query_text.strip():
        return []

    cleaned_query = query_text.strip()
    tsquery = func.websearch_to_tsquery(TEXT_SEARCH_CONFIG, cleaned_query)

    base_query = (
        db.query(DocumentChunk)
        .options(joinedload(DocumentChunk.document))
        .join(SourceDocument, DocumentChunk.source_document_id == SourceDocument.id)
        .filter(DocumentChunk.search_vector.op("@@")(tsquery))
    )
    filtered_query = _apply_document_filters(
        base_query, user_id=user_id, document_ids=document_ids, filenames=filenames
    )

    results = (
        filtered_query.order_by(func.ts_rank(DocumentChunk.search_vector, tsquery).desc())
        .limit(limit)
        .all()
    )

    # Fallback if websearch returns empty on highly specific punctuation/phrasing
    if not results:
        plain_query = func.plainto_tsquery(TEXT_SEARCH_CONFIG, cleaned_query)
        fallback_base = (
            db.query(DocumentChunk)
            .options(joinedload(DocumentChunk.document))
            .join(SourceDocument, DocumentChunk.source_document_id == SourceDocument.id)
            .filter(DocumentChunk.search_vector.op("@@")(plain_query))
        )
        fallback_filtered = _apply_document_filters(
            fallback_base, user_id=user_id, document_ids=document_ids, filenames=filenames
        )

        results = (
            fallback_filtered.order_by(
                func.ts_rank(DocumentChunk.search_vector, plain_query).desc()
            )
            .limit(limit)
            .all()
        )

    return results


def document_chunk_to_langchain_document(
    chunk: DocumentChunk,
    score: float | None = None,
) -> Document:
    """
    Transforms a raw database DocumentChunk into a canonical LangChain Document
    with strict document provenance for citations and multi-document comparison.
    """
    filename = (
        chunk.document.filename
        if getattr(chunk, "document", None) and getattr(chunk.document, "filename", None)
        else "unknown.pdf"
    )

    metadata: dict[str, Any] = {
        "source": filename,
        "page": chunk.page_number,
        "chunk_index": chunk.chunk_index,
        "section_name": chunk.section_name or "General",
        "chunk_id": str(chunk.id),
        "document_id": str(chunk.source_document_id),
    }
    if score is not None:
        metadata["score"] = score

    return Document(page_content=chunk.text_content, metadata=metadata)


def semantic_search_documents(
    db: Session,
    query_embedding: list[float],
    user_id: UUID | None = None,
    document_ids: Sequence[UUID] | None = None,
    filenames: Sequence[str] | None = None,
    limit: int = 20,
) -> list[Document]:
    """
    Executes pgvector semantic similarity search and directly yields
    canonical LangChain Document objects.
    """
    chunks = semantic_search(
        db=db,
        query_embedding=query_embedding,
        user_id=user_id,
        document_ids=document_ids,
        filenames=filenames,
        limit=limit,
    )
    return [document_chunk_to_langchain_document(c) for c in chunks]


def full_text_search_documents(
    db: Session,
    query_text: str,
    user_id: UUID | None = None,
    document_ids: Sequence[UUID] | None = None,
    filenames: Sequence[str] | None = None,
    limit: int = 20,
) -> list[Document]:
    """
    Executes PostgreSQL full-text search and directly yields
    canonical LangChain Document objects.
    """
    chunks = full_text_search(
        db=db,
        query_text=query_text,
        user_id=user_id,
        document_ids=document_ids,
        filenames=filenames,
        limit=limit,
    )
    return [document_chunk_to_langchain_document(c) for c in chunks]
