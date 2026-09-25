"""
Upload ingestion service: handles streaming PDF bytes -> parse -> chunk -> embed -> save.
Reuses the existing parser/chunker/embedder pipeline from Phase 0–5 ingest.
"""

import time
import logging
from uuid import UUID

from sqlalchemy.orm import Session

from app.database.models.source_document import SourceDocument
from app.database.models.document_chunk import DocumentChunk
from app.ingest.chunker import chunk_page
from app.ingest.embedder import get_embeddings
from app.ingest.parser import parse_pdf_bytes

logger = logging.getLogger(__name__)


def ingest_uploaded_document(
    db: Session,
    user_id: UUID,
    filename: str,
    file_bytes: bytes,
) -> SourceDocument:
    """
    Parse, chunk, embed, and persist a user-uploaded PDF.
    Returns the saved SourceDocument instance.

    Raises ValueError for non-text PDFs (zero extractable content).
    Raises RuntimeError on any upstream embedding failure.
    """
    logger.info(f"Starting upload ingestion for '{filename}' user={user_id}")

    pages = parse_pdf_bytes(file_bytes)
    logger.info(f"Extracted {len(pages)} non-empty pages from upload '{filename}'")

    if not pages:
        raise ValueError(f"No extractable text found in '{filename}'. Cannot ingest.")

    all_chunks = []
    header_stack: list[tuple[int, str]] = []
    chunk_idx = 0

    for page in pages:
        page_chunks, header_stack = chunk_page(
            page_text=page["text"],
            page_number=page["page_number"],
            start_chunk_idx=chunk_idx,
            header_stack=header_stack,
            chunk_size=1000,
            chunk_overlap=200,
        )
        all_chunks.extend(page_chunks)
        chunk_idx += len(page_chunks)

    logger.info(f"Chunked '{filename}': {len(all_chunks)} chunks. Embedding...")

    batch_size = 20
    embeddings: list = []
    for i in range(0, len(all_chunks), batch_size):
        if i > 0:
            time.sleep(3.0)
        chunk_batch = all_chunks[i : i + batch_size]
        batch_texts = [c["text_content"] for c in chunk_batch]
        batch_prefixes = [
            f"Document: {filename} | Section: {c['section_name'] or 'General'}"
            for c in chunk_batch
        ]
        embeddings.extend(get_embeddings(batch_texts, prefixes=batch_prefixes))

    logger.info(f"Embedding complete for '{filename}'. Saving to DB...")

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
        for i, chunk in enumerate(all_chunks)
    ]
    db.bulk_save_objects(db_chunks)
    db.commit()
    db.refresh(db_doc)

    logger.info(f"Saved '{filename}' ({len(db_chunks)} chunks) for user={user_id}")
    return db_doc
