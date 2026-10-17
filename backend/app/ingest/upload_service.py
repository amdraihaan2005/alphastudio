"""
Upload Ingestion Service: Canonical application service layer for PDF document persistence.

Architecture (Two-Phase Decoupled Ingestion):
- Phase 1 (Data & AI Layer):
  Delegates CPU-bound extraction, structure-aware chunking, and concurrent AI vector embedding
  to `orchestrator.process_pdf_to_embeddings` BEFORE touching the database.
  Zero database connections, sessions, or row locks are held during CPU parsing or Cohere network requests.
- Phase 2 (ACID Relational & Vector Persistence):
  Encapsulates relational record creation, foreign key mapping, and pgvector persistence
  inside a sub-second atomic transaction with explicit rollback on failure.
"""

from collections.abc import Sequence
import logging
from uuid import UUID

from langchain_core.documents import Document
from sqlalchemy.orm import Session

from app.database.models.document_chunk import DocumentChunk
from app.database.models.source_document import SourceDocument
from app.ingest.orchestrator import (
    LangChainIngestOrchestrator,
    default_orchestrator,
    process_pdf_to_embeddings,
)

logger = logging.getLogger(__name__)


class UploadIngestionService:
    """
    Application service managing the complete lifecycle of user-uploaded document ingestion:
    Orchestration -> Validation -> Relational & Vector Persistence.
    """

    def __init__(
        self,
        orchestrator: LangChainIngestOrchestrator | None = None,
    ) -> None:
        self.orchestrator = orchestrator or default_orchestrator

    def build_chunk_records(
        self,
        document_id: UUID,
        chunks: Sequence[Document],
        embeddings: list[list[float]],
    ) -> list[DocumentChunk]:
        """
        Maps LangChain Document chunks and dense vector embeddings into relational
        DocumentChunk models with strict validation against length mismatches.
        """
        if len(chunks) != len(embeddings):
            raise RuntimeError(
                f"Embedding count ({len(embeddings)}) does not match chunk count ({len(chunks)})."
            )

        db_chunks: list[DocumentChunk] = []
        for idx, (doc, emb) in enumerate(zip(chunks, embeddings)):
            chunk_idx = doc.metadata.get("chunk_index", idx)
            page_num = doc.metadata.get("page", 1)
            section = doc.metadata.get("section_name")

            db_chunks.append(
                DocumentChunk(
                    source_document_id=document_id,
                    chunk_index=chunk_idx,
                    page_number=page_num,
                    section_name=section,
                    text_content=doc.page_content,
                    embedding=emb,
                )
            )
        return db_chunks

    async def ingest(
        self,
        db: Session,
        user_id: UUID,
        filename: str,
        file_bytes: bytes,
        ticker: str = "USER_UPLOAD",
        filing_type: str = "CUSTOM",
        year: int = 0,
    ) -> SourceDocument:
        """
        Parse, chunk, embed, and persist a user-uploaded PDF document.

        PHASE 1 (Data & AI):
          Offloads extraction, chunking, and vector embedding to the LangChain orchestrator
          prior to opening any database transaction.

        PHASE 2 (ACID DB Transaction):
          Encapsulates database insertion, foreign key mapping, and commit
          inside a short-lived transaction with explicit rollback on error.
        """
        logger.info(
            f"Starting upload ingestion for '{filename}' user={user_id}"
        )

        # PHASE 1: Offload extraction, chunking, and AI embedding to orchestrator
        # Calls process_pdf_to_embeddings at module level to allow seamless mocking in unit tests
        chunks, embeddings = await process_pdf_to_embeddings(
            file_bytes, filename
        )

        if not chunks:
            raise ValueError(
                f"No extractable text found in '{filename}'. Cannot ingest."
            )

        if len(chunks) != len(embeddings):
            raise RuntimeError(
                f"Embedding count ({len(embeddings)}) does not match chunk count ({len(chunks)}) for '{filename}'."
            )

        logger.info(
            f"Embedding complete for '{filename}' ({len(chunks)} chunks). Persisting to database..."
        )

        # PHASE 2: ACID Database Transaction strictly wrapping SQL operations
        try:
            db_doc = SourceDocument(
                filename=filename,
                user_id=user_id,
                ticker=ticker,
                filing_type=filing_type,
                year=year,
            )
            db.add(db_doc)
            db.flush()

            db_chunks = self.build_chunk_records(
                document_id=db_doc.id,
                chunks=chunks,
                embeddings=embeddings,
            )
            db.bulk_save_objects(db_chunks)
            db.commit()
            db.refresh(db_doc)

            logger.info(
                f"Successfully saved '{filename}' ({len(db_chunks)} chunks) for user={user_id}"
            )
            return db_doc

        except Exception as e:
            db.rollback()
            logger.error(
                f"Database transaction failed for '{filename}', rolled back: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Database error during document ingestion for '{filename}': {e}"
            ) from e


default_upload_service = UploadIngestionService()


async def ingest_uploaded_document(
    db: Session,
    user_id: UUID,
    filename: str,
    file_bytes: bytes,
    ticker: str = "USER_UPLOAD",
    filing_type: str = "CUSTOM",
    year: int = 0,
) -> SourceDocument:
    """
    Public entrypoint for ingesting user-uploaded documents.
    Delegates directly to default_upload_service.ingest.
    """
    return await default_upload_service.ingest(
        db=db,
        user_id=user_id,
        filename=filename,
        file_bytes=file_bytes,
        ticker=ticker,
        filing_type=filing_type,
        year=year,
    )
