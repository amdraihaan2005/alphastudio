"""
Upload ingestion service: coordinates PDF ingestion as an application service layer (traffic cop).
Delegates CPU-bound extraction and AI vector embedding to the async orchestrator,
and wraps persistence strictly inside a short-lived ACID database transaction with rollback.
"""

import logging
from uuid import UUID

from sqlalchemy.orm import Session

from app.database.models.document_chunk import DocumentChunk
from app.database.models.source_document import SourceDocument
from app.ingest.orchestrator import process_pdf_to_embeddings

logger = logging.getLogger(__name__)


async def ingest_uploaded_document(
    db: Session,
    user_id: UUID,
    filename: str,
    file_bytes: bytes,
) -> SourceDocument:
    """
    Parse, chunk, embed, and persist a user-uploaded PDF.

    PHASE 1 (Data & AI):
      Delegates parsing, chunking, enriched prefixing, and Cohere embeddings
      to the asynchronous orchestrator *before* touching the database.
      Zero database sessions or connections are held during CPU/API execution.

    PHASE 2 (ACID DB Transaction):
      Strictly encapsulates database insertion, foreign key mapping, and commit
      inside a millisecond-duration transaction with explicit rollback on error.
    """
    logger.info(f"Starting upload ingestion for '{filename}' user={user_id}")

    # PHASE 1: Offload extraction, chunking, and AI embedding to orchestrator
    chunks, embeddings = await process_pdf_to_embeddings(file_bytes, filename)

    if not chunks:
        raise ValueError(f"No extractable text found in '{filename}'. Cannot ingest.")

    logger.info(
        f"Embedding complete for '{filename}' ({len(chunks)} chunks). Persisting to database..."
    )

    # PHASE 2: ACID Database Transaction strictly wrapping SQL operations
    try:
        db_doc = SourceDocument(
            filename=filename,
            user_id=user_id,
            ticker="USER_UPLOAD",
            filing_type="CUSTOM",
            year=0,
        )
        db.add(db_doc)
        db.flush()

        db_chunks = [
            DocumentChunk(
                source_document_id=db_doc.id,
                chunk_index=chunk["chunk_index"],
                page_number=chunk["page_number"],
                section_name=chunk["section_name"],
                text_content=chunk["text_content"],
                embedding=embeddings[i],
            )
            for i, chunk in enumerate(chunks)
        ]
        db.bulk_save_objects(db_chunks)
        db.commit()
        db.refresh(db_doc)

        logger.info(f"Successfully saved '{filename}' ({len(db_chunks)} chunks) for user={user_id}")
        return db_doc

    except Exception as e:
        db.rollback()
        logger.error(f"Database transaction failed for '{filename}', rolled back: {e}", exc_info=True)
        raise RuntimeError(f"Database error during document ingestion for '{filename}': {e}") from e
