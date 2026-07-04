# ruff: noqa: E402
import os
import json
import logging
import sys
import time
from sqlalchemy.orm import Session

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("app.ingest.pipeline")

from app.database.connection import SessionLocal
from app.database.models.source_document import SourceDocument
from app.database.models.document_chunk import DocumentChunk
from app.ingest.parser import parse_pdf_pages
from app.ingest.chunker import chunk_page
from app.ingest.embedder import get_embeddings

def ingest_document(db: Session, doc_metadata: dict, pdf_file_path: str) -> None:
    """
    Ingests a single annual report PDF document.
    Parses, chunks, embeds, and saves elements atomically to the database.
    """
    filename = doc_metadata["filename"]
    ticker = doc_metadata["ticker"]
    company = doc_metadata["company"]
    year = doc_metadata["year"]
    filing_type = doc_metadata["type"]
    
    logger.info(f"Starting ingestion process for {company} ({year}) - {filename}")
    
    # 1. Parse pages page-by-page (keeps memory low)
    pages_generator = parse_pdf_pages(pdf_file_path)
    
    all_chunks = []
    current_section = None
    chunk_idx = 0
    
    try:
        for page in pages_generator:
            page_text = page["text"]
            page_num = page["page_number"]
            
            if not page_text.strip():
                continue
                
            # Chunk the page text
            page_chunks, current_section = chunk_page(
                page_text=page_text,
                page_number=page_num,
                start_chunk_idx=chunk_idx,
                current_section=current_section,
                chunk_size=1000,
                chunk_overlap=200
            )
            
            all_chunks.extend(page_chunks)
            chunk_idx += len(page_chunks)
            
        logger.info(f"Successfully chunked {filename}. Total chunks generated: {len(all_chunks)}")
        
        if not all_chunks:
            logger.warning(f"No text extracted from document {filename}. Skipping insertion.")
            return
            
        # 2. Batch embedding calls in chunks of 20 to stay within Cohere trial token rate limits
        batch_size = 20
        embeddings = []
        
        for i in range(0, len(all_chunks), batch_size):
            # Proactive rate limit spacing: sleep 3 seconds between batches (except the first one)
            if i > 0:
                time.sleep(3.0)
                
            chunk_batch = all_chunks[i : i + batch_size]
            batch_texts = [c["text_content"] for c in chunk_batch]
            
            batch_embeddings = get_embeddings(batch_texts)
            embeddings.extend(batch_embeddings)
            
        logger.info(f"Successfully generated embeddings for all {len(all_chunks)} chunks.")
        
        # 3. Create parent SourceDocument and its children DocumentChunks and save atomically
        # Start of transaction (using nested or direct session transaction)
        db_doc = SourceDocument(
            filename=filename,
            ticker=ticker,
            filing_type=filing_type,
            year=year
        )
        db.add(db_doc)
        db.flush() # Flushes to database to populate db_doc.id primary key
        
        db_chunks = []
        for i, chunk in enumerate(all_chunks):
            db_chunk = DocumentChunk(
                source_document_id=db_doc.id,
                chunk_index=chunk["chunk_index"],
                page_number=chunk["page_number"],
                section_name=chunk["section_name"],
                text_content=chunk["text_content"],
                embedding=embeddings[i]
            )
            db_chunks.append(db_chunk)
            
        # Bulk save for high insertion performance
        db.bulk_save_objects(db_chunks)
        db.commit()
        
        logger.info(f"[+] Successfully saved {filename} and all {len(db_chunks)} chunks to Supabase.")
        
    except Exception as e:
        db.rollback()
        logger.error(f"[-] Failed ingestion for {filename}. Database transaction rolled back. Error: {e}", exc_info=True)
        raise e

def run_pipeline() -> None:
    """
    Main orchestrator that reads manifest.json and loops through all reports.
    """
    # Find directories relative to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    downloads_dir = os.path.join(base_dir, "data", "downloads")
    manifest_path = os.path.join(downloads_dir, "manifest.json")
    
    if not os.path.exists(manifest_path):
        logger.error(f"manifest.json not found at {manifest_path}. Please check Phase 0 configuration.")
        return
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    logger.info(f"Found {len(manifest)} documents listed in manifest.json")
    
    db = SessionLocal()
    try:
        for doc in manifest:
            filename = doc["filename"]
            pdf_path = os.path.join(downloads_dir, filename)
            
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file does not exist at {pdf_path}. Skipping.")
                continue
                
            # Idempotency check: verify if document is already ingested
            existing_doc = db.query(SourceDocument).filter(SourceDocument.filename == filename).first()
            if existing_doc:
                logger.info(f"Idempotency Guard: Document '{filename}' is already ingested (ID: {existing_doc.id}). Skipping.")
                continue
                
            ingest_document(db, doc, pdf_path)
            
    finally:
        db.close()

if __name__ == "__main__":
    run_pipeline()
