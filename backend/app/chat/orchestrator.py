import asyncio
import re
import uuid
from uuid import UUID

from starlette.concurrency import run_in_threadpool
import structlog

from app.assistant.agent import DocumentAgentDeps, agent
from app.chat.streaming import (
    format_data_part,
    format_end_part,
    format_error_part,
    format_finish_part,
    format_start_part,
    format_text_part,
)
from app.database.connection import SessionLocal
from app.database.models.chat_message import ChatMessage
from app.database.models.chat_thread import ChatThread
from app.database.models.message_citation import MessageCitation
from app.grounding.validator import GroundingValidationError, validate_citations
from app.retrieval.retriever import retrieve_hybrid

logger = structlog.get_logger(__name__)


def _save_message_and_citations(
    db, assistant_msg_id: UUID, thread_id: UUID, full_text: str, validated_citations: list
):
    """Persists the assistant message and verified citations to the database."""
    db_msg = ChatMessage(
        id=assistant_msg_id,
        chat_thread_id=thread_id,
        role="assistant",
        content=full_text,
    )
    db.add(db_msg)

    for cit in validated_citations:
        db_cit = MessageCitation(message_id=assistant_msg_id, chunk_id=cit["chunk_id"])
        db.add(db_cit)

    db.commit()


async def orchestrate_chat_stream(
    thread_id: UUID, user_query: str, user_id: UUID | None = None
):
    """
    Orchestrates a genuinely grounded, non-blocking RAG chat turn:
    1. Runs hybrid search in a threadpool to avoid blocking the asyncio event loop.
    2. Yields retrieved chunks metadata to the client.
    3. Executes the agent and intercepts the response BEFORE streaming to the user.
    4. Enforces strict fail-closed citation validation: ungrounded text is suppressed.
    5. Streams only verified, grounded text to the client with smooth token delivery.
    6. Persists message and citations atomically in a threadpool.
    """
    logger.info("Orchestrating chat stream", thread_id=str(thread_id), query=user_query)

    db = SessionLocal()
    assistant_msg_id = uuid.uuid4()
    assistant_msg_id_str = str(assistant_msg_id)

    try:
        # Resolve user_id if not provided without blocking event loop
        if user_id is None:
            thread = await run_in_threadpool(
                lambda: db.query(ChatThread).filter(ChatThread.id == thread_id).first()
            )
            if thread:
                user_id = thread.user_id

        # Non-blocking hybrid retrieval
        retrieved_chunks = await run_in_threadpool(
            retrieve_hybrid,
            db,
            user_query,
            user_id=user_id,
            limit=5,
            fetch_neighbors=True,
        )

        retrieved_metadata = []
        for idx, r in enumerate(retrieved_chunks):
            c = r["chunk"]
            retrieved_metadata.append(
                {
                    "id": str(c.id),
                    "filename": c.document.filename,
                    "page_number": c.page_number,
                    "section_name": c.section_name,
                    "score": r["score"],
                }
            )
        yield format_data_part(
            {"chunks": retrieved_metadata}, "retrieved_context", assistant_msg_id_str
        )

        deps = DocumentAgentDeps(
            db=db, retrieved_chunks=retrieved_chunks, user_id=user_id
        )

        # Run agent to generate candidate response
        agent_result = await agent.run(user_query, deps=deps)
        candidate_text = str(getattr(agent_result, "output", getattr(agent_result, "data", agent_result)))

        # Normalize unicode and brackets for robust citation extraction
        normalized_text = candidate_text.replace("【", "[").replace("】", "]")
        normalized_text = re.sub(
            r"[\u200b\u200f\u202f\u00a0\u2002\u2003\u2009]", " ", normalized_text
        )

        # STRICT FAIL-CLOSED VALIDATION:
        # If citations fail validation, GroundingValidationError is raised HERE,
        # BEFORE any token has been emitted to the user's screen!
        validated_citations = validate_citations(normalized_text, deps.retrieved_chunks)

        # Persist to database in threadpool
        await run_in_threadpool(
            _save_message_and_citations,
            db,
            assistant_msg_id,
            thread_id,
            normalized_text,
            validated_citations,
        )

        logger.info(
            "Successfully saved assistant message and citations to database",
            message_id=str(assistant_msg_id),
            citations_count=len(validated_citations),
        )

        # Stream the 100% verified, grounded text to the client
        yield format_start_part(assistant_msg_id_str)

        # Stream words/tokens smoothly for UI typing experience
        words = re.findall(r"\S+|\s+", normalized_text)
        batch = []
        for word in words:
            batch.append(word)
            if len(batch) >= 3:
                delta = "".join(batch)
                yield format_text_part(delta, assistant_msg_id_str)
                batch = []
                await asyncio.sleep(0.01)

        if batch:
            yield format_text_part("".join(batch), assistant_msg_id_str)

        yield format_end_part(assistant_msg_id_str)

        # Emit validated citation metadata
        yield format_data_part(
            {
                "citations": [
                    {
                        "filename": c["filename"],
                        "page_number": c["page_number"],
                        "chunk_id": str(c["chunk_id"]),
                    }
                    for c in validated_citations
                ]
            },
            "citations",
            assistant_msg_id_str,
        )

        yield format_finish_part()

    except GroundingValidationError as validation_err:
        await run_in_threadpool(db.rollback)
        logger.warning(
            "Grounding violation intercepted before rendering", error=str(validation_err)
        )
        # Suppress hallucination and deliver genuine fail-closed message
        yield format_start_part(assistant_msg_id_str)
        refusal_msg = (
            f"**Grounding Protection Alert**: The draft answer cited sources that could not be verified in the retrieved documents:\n\n"
            f"> *{str(validation_err)}*\n\n"
            f"To guarantee institutional accuracy, ungrounded claims are blocked. Please refine your query or check available filings."
        )
        yield format_text_part(refusal_msg, assistant_msg_id_str)
        yield format_end_part(assistant_msg_id_str)
        yield format_finish_part()

    except Exception as err:
        await run_in_threadpool(db.rollback)
        logger.error(
            "Error during orchestrate_chat_stream", error=str(err), exc_info=True
        )
        yield format_error_part(f"Error: {str(err)}")
    finally:
        await run_in_threadpool(db.close)
