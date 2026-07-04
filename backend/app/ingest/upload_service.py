"""
Upload ingestion service: handles streaming PDF bytes -> parse -> chunk -> embed -> save.
Reuses the existing parser/chunker/embedder pipeline from Phase 0–5 ingest.
"""
import io
import time
import logging
from uuid import UUID

from sqlalchemy.orm import Session

from app.database.models.source_document import SourceDocument
from app.database.models.document_chunk import DocumentChunk
from app.ingest.chunker import chunk_page
from app.ingest.embedder import get_embeddings

logger = logging.getLogger(__name__)


def parse_pdf_bytes(file_bytes: bytes):
    """
    Parse a PDF from raw bytes using PyMuPDF (fitz).
    Yields ParsedPage dicts with page_number and text.
    """
    import fitz  # PyMuPDF
    from app.ingest.parser import table_to_markdown, is_overlapping

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        for page_idx, page in enumerate(doc):
            page_number = page_idx + 1

            # Identify tables and convert to Markdown
            tables = page.find_tables()
            table_bboxes = []
            table_markdowns = {}
            for t in tables.tables:
                bbox = t.bbox
                table_bboxes.append(bbox)
                try:
                    table_markdowns[bbox] = table_to_markdown(t.extract())
                except Exception:
                    pass

            # Extract text blocks, skipping text inside tables
            blocks = page.get_text("blocks")
            elements = []
            for bbox, md in table_markdowns.items():
                elements.append((bbox[1], bbox[0], "table", md, bbox))
            for block in blocks:
                block_bbox = block[:4]
                text = block[4].strip()
                if not text:
                    continue
                in_table = any(is_overlapping(block_bbox, t_bbox, 0.4) for t_bbox in table_bboxes)
                if not in_table:
                    elements.append((block_bbox[1], block_bbox[0], "text", text, block_bbox))

            elements.sort(key=lambda x: (x[0], x[1]))
            assembled = "\n\n".join(e[3] for e in elements).strip()
            yield {"page_number": page_number, "text": assembled}
    finally:
        doc.close()


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

    all_chunks = []
    current_section = None
    chunk_idx = 0

    for page in parse_pdf_bytes(file_bytes):
        if not page["text"].strip():
            continue
        page_chunks, current_section = chunk_page(
            page_text=page["text"],
            page_number=page["page_number"],
            start_chunk_idx=chunk_idx,
            current_section=current_section,
            chunk_size=1000,
            chunk_overlap=200,
        )
        all_chunks.extend(page_chunks)
        chunk_idx += len(page_chunks)

    if not all_chunks:
        raise ValueError(f"No extractable text found in '{filename}'. Cannot ingest.")

    logger.info(f"Chunked '{filename}': {len(all_chunks)} chunks. Embedding...")

    # Batch embed (respecting Cohere trial rate limits)
    batch_size = 20
    embeddings: list = []
    for i in range(0, len(all_chunks), batch_size):
        if i > 0:
            time.sleep(3.0)
        batch_texts = [c["text_content"] for c in all_chunks[i : i + batch_size]]
        embeddings.extend(get_embeddings(batch_texts))

    logger.info(f"Embedding complete for '{filename}'. Saving to DB...")

    # Persist atomically
    db_doc = SourceDocument(
        filename=filename,
        user_id=user_id,
        ticker="USER_UPLOAD",
        filing_type="CUSTOM",
        year=0,
    )
    db.add(db_doc)
    db.flush()  # Populate db_doc.id

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
