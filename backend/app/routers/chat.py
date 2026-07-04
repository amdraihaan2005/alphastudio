from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from uuid import UUID
from typing import List
import asyncio

from app.database.connection import get_db
from app.database.models.chat_thread import ChatThread
from app.database.models.chat_message import ChatMessage
from app.database.models.document_chunk import DocumentChunk
from app.database.models.source_document import SourceDocument
from app.database.models.user import User
from app.auth.dependencies import get_current_user
from app.schemas.chat import ThreadCreate, ThreadResponse, MessageResponse, ChatStreamRequest
from app.chat.orchestrator import orchestrate_chat_stream

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.get("/threads", response_model=List[ThreadResponse])
def list_threads(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List all chat threads belonging to the current authenticated user.
    """
    threads = db.query(ChatThread).filter(ChatThread.user_id == current_user.id).order_by(ChatThread.created_at.desc()).all()
    return threads

@router.post("/threads", response_model=ThreadResponse, status_code=status.HTTP_201_CREATED)
def create_thread(
    thread_data: ThreadCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new chat thread for the current authenticated user.
    """
    new_thread = ChatThread(
        user_id=current_user.id,
        title=thread_data.title
    )
    db.add(new_thread)
    db.commit()
    db.refresh(new_thread)
    return new_thread

@router.get("/threads/{thread_id}/messages", response_model=List[MessageResponse])
def get_thread_messages(
    thread_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Load message history for a specific chat thread.
    Validates ownership of the thread first, raising 403 Forbidden if accessed by another user.
    """
    thread = db.query(ChatThread).filter(ChatThread.id == thread_id).first()
    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat thread not found"
        )
    
    if thread.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this chat thread"
        )
        
    return thread.messages

@router.delete("/threads/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_thread(
    thread_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a chat thread and all its associated messages.
    Validates ownership of the thread first, raising 403 Forbidden if accessed by another user.
    """
    thread = db.query(ChatThread).filter(ChatThread.id == thread_id).first()
    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat thread not found"
        )
        
    if thread.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this chat thread"
        )
        
    db.delete(thread)
    db.commit()
    return


@router.post("/stream")
async def chat_stream(
    request: ChatStreamRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Accepts the chat message history, performs hybrid retrieval, invokes PydanticAI
    agent, runs grounding check, and streams response in Vercel AI SDK compatible format.
    """
    # 1. Validate ownership of thread
    thread = db.query(ChatThread).filter(ChatThread.id == request.thread_id).first()
    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat thread not found"
        )
        
    if thread.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this chat thread"
        )
        
    if not request.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message history cannot be empty"
        )
        
    # Get the last message (the new user message)
    new_user_msg = request.messages[-1]
    if new_user_msg.role != "user":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last message in history must be from user"
        )
        
    # 2. Persist the user's message immediately
    user_msg_db = ChatMessage(
        chat_thread_id=request.thread_id,
        role="user",
        content=new_user_msg.content
    )
    db.add(user_msg_db)
    db.commit()
    
    # 3. Stream from orchestrator
    return StreamingResponse(
        orchestrate_chat_stream(request.thread_id, new_user_msg.content),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no"
        }
    )

@router.get("/chunks/{chunk_id}")
def get_chunk_details(
    chunk_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve document chunk details along with context from surrounding chunks.
    """
    chunk = db.query(DocumentChunk).filter(DocumentChunk.id == chunk_id).first()
    if not chunk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document chunk not found"
        )

    # Retrieve preceding and succeeding chunks
    preceding_chunk = db.query(DocumentChunk).filter(
        DocumentChunk.source_document_id == chunk.source_document_id,
        DocumentChunk.chunk_index == chunk.chunk_index - 1
    ).first()

    succeeding_chunk = db.query(DocumentChunk).filter(
        DocumentChunk.source_document_id == chunk.source_document_id,
        DocumentChunk.chunk_index == chunk.chunk_index + 1
    ).first()

    return {
        "id": str(chunk.id),
        "text_content": chunk.text_content,
        "page_number": chunk.page_number,
        "section_name": chunk.section_name,
        "document": {
            "id": str(chunk.document.id),
            "filename": chunk.document.filename,
            "ticker": chunk.document.ticker,
            "filing_type": chunk.document.filing_type,
            "year": chunk.document.year,
        },
        "preceding_text": preceding_chunk.text_content if preceding_chunk else None,
        "succeeding_text": succeeding_chunk.text_content if succeeding_chunk else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 10: Private PDF Upload
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Accept a user-uploaded PDF, parse it, chunk it, embed via Cohere, and save
    as a private SourceDocument scoped to current_user.id.

    The heavy ingestion work is offloaded to a background task so the HTTP
    response returns immediately with 202 Accepted.
    """
    # Validate MIME type
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported.",
        )

    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File must have a .pdf extension.",
        )

    # Read bytes eagerly (before request scope closes)
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    user_id = current_user.id

    # Idempotency guard: reject if the user already has a doc with the same filename
    existing = (
        db.query(SourceDocument)
        .filter(
            SourceDocument.filename == filename,
            SourceDocument.user_id == user_id,
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"A document named '{filename}' already exists in your library. "
                "Please delete it first or rename the file before re-uploading."
            ),
        )

    def _run_ingestion():
        from app.database.connection import SessionLocal
        from app.ingest.upload_service import ingest_uploaded_document
        import logging

        logger = logging.getLogger(__name__)
        session = SessionLocal()
        try:
            ingest_uploaded_document(session, user_id, filename, file_bytes)
        except Exception as exc:
            logger.error(f"Background ingestion failed for '{filename}': {exc}", exc_info=True)
        finally:
            session.close()

    background_tasks.add_task(_run_ingestion)

    return {
        "status": "processing",
        "filename": filename,
        "message": "Your document is being ingested. It will be available for chat shortly.",
    }


@router.get("/documents/public")
def list_public_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Return all pre-loaded public documents (user_id IS NULL).
    These are system-level filings available to every user and cannot be deleted.
    """
    docs = (
        db.query(SourceDocument)
        .filter(SourceDocument.user_id.is_(None))
        .order_by(SourceDocument.filename)
        .all()
    )
    return [
        {
            "id": str(d.id),
            "filename": d.filename,
            "ticker": d.ticker,
            "year": d.year,
        }
        for d in docs
    ]


@router.get("/documents")
def list_user_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Return all private documents uploaded by the current user.
    """
    docs = (
        db.query(SourceDocument)
        .filter(SourceDocument.user_id == current_user.id)
        .order_by(SourceDocument.created_at.desc())
        .all()
    )
    return [
        {
            "id": str(d.id),
            "filename": d.filename,
            "created_at": d.created_at.isoformat(),
        }
        for d in docs
    ]


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user_document(
    doc_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Delete a private document uploaded by the current user.
    Cascades to all associated DocumentChunks and MessageCitations.
    Only the owning user may delete their own documents.
    """
    doc = (
        db.query(SourceDocument)
        .filter(
            SourceDocument.id == doc_id,
            SourceDocument.user_id == current_user.id,
        )
        .first()
    )
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or you do not have permission to delete it.",
        )
    db.delete(doc)
    db.commit()
    return
