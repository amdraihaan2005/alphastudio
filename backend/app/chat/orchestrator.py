import structlog
import uuid
from uuid import UUID

from app.database.connection import SessionLocal
from app.database.models.chat_message import ChatMessage
from app.database.models.chat_thread import ChatThread
from app.database.models.message_citation import MessageCitation
from app.retrieval.retriever import retrieve_hybrid
from app.assistant.agent import agent, DocumentAgentDeps
from app.grounding.validator import validate_citations, GroundingValidationError
from app.chat.streaming import (
  format_start_part,
  format_text_part,
  format_end_part,
  format_finish_part,
  format_data_part,
  format_error_part
)

logger = structlog.get_logger(__name__)

async def orchestrate_chat_stream(thread_id: UUID, user_query: str, user_id: UUID | None = None):
    """
    Orchestrates a single RAG chat turn:
    1. Runs hybrid search to retrieve context.
    2. Yields retrieved chunks immediately as a metadata part.
    3. Invokes the PydanticAI agent to stream the response.
    4. Validates that generated inline citations are grounded in retrieved context.
    5. Saves the assistant's message and citations atomically to the database.
    """
    logger.info("Orchestrating chat stream", thread_id=str(thread_id), query=user_query)

    
    # Open a single database session for the entire lifecycle of this turn
    db = SessionLocal()
    assistant_msg_id = uuid.uuid4()
    assistant_msg_id_str = str(assistant_msg_id)
    
    try:
        # Resolve user_id from thread if not provided
        if user_id is None:
            thread = db.query(ChatThread).filter(ChatThread.id == thread_id).first()
            if thread:
                user_id = thread.user_id

        # 1. Retrieve context chunks (scoped to user)
        retrieved_chunks = retrieve_hybrid(db, user_query, user_id=user_id, limit=5, fetch_neighbors=True)
        
        # Send metadata about retrieved chunks back to the client immediately
        retrieved_metadata = []
        for idx, r in enumerate(retrieved_chunks):
            c = r["chunk"]
            retrieved_metadata.append({
                "id": str(c.id),
                "filename": c.document.filename,
                "page_number": c.page_number,
                "section_name": c.section_name,
                "score": r["score"]
            })
        yield format_data_part({"chunks": retrieved_metadata}, "retrieved_context", assistant_msg_id_str)
        
        # 2. Invoke PydanticAI agent and stream response
        deps = DocumentAgentDeps(db=db, retrieved_chunks=retrieved_chunks, user_id=user_id)
        full_text = ""
        
        # Yield the text start part
        yield format_start_part(assistant_msg_id_str)
        
        # PydanticAI stream context
        async with agent.run_stream(user_query, deps=deps) as result:
            async for text_delta in result.stream_text(delta=True):
                full_text += text_delta
                yield format_text_part(text_delta, assistant_msg_id_str)
                
        # Yield the text end part
        yield format_end_part(assistant_msg_id_str)
        
        # 3. Post-stream: Validate citations and persist to DB atomically
        import re
        full_text = full_text.replace("【", "[").replace("】", "]")
        full_text = re.sub(r"[\u200b\u200f\u202f\u00a0\u2002\u2003\u2009]", " ", full_text)
        
        validated_citations = validate_citations(full_text, retrieved_chunks)
        
        # Persist assistant's message
        db_msg = ChatMessage(
            id=assistant_msg_id,
            chat_thread_id=thread_id,
            role="assistant",
            content=full_text
        )
        db.add(db_msg)
        
        # Persist associated citations links
        for cit in validated_citations:
            db_cit = MessageCitation(
                message_id=assistant_msg_id,
                chunk_id=cit["chunk_id"]
            )
            db.add(db_cit)
            
        db.commit()
        logger.info("Successfully saved assistant message and citations to database", message_id=str(assistant_msg_id), citations_count=len(validated_citations))
        
        # Yield the validated citations metadata block
        yield format_data_part({
            "citations": [
                {"filename": c["filename"], "page_number": c["page_number"], "chunk_id": str(c["chunk_id"])} 
                for c in validated_citations
            ]
        }, "citations", assistant_msg_id_str)
        
        # Yield finish step to notify completion
        yield format_finish_part()
        
    except GroundingValidationError as validation_err:
        db.rollback()
        logger.warning("Grounding violation blocked response", error=str(validation_err))
        yield format_error_part(f"Grounding Error: {str(validation_err)}")
    except Exception as err:
        db.rollback()
        logger.error("Error during orchestrate_chat_stream", error=str(err), exc_info=True)
        yield format_error_part(f"Error: {str(err)}")
    finally:
        db.close()

