import os
import logging
from uuid import UUID
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from sqlalchemy.orm import Session

from app.config import settings
from app.database.models.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)

# Define dependency structure passed to agent run context
class DocumentAgentDeps:
    def __init__(self, db: Session, retrieved_chunks: list[dict], user_id=None):
        self.db = db
        self.retrieved_chunks = retrieved_chunks
        self.user_id = user_id  # UUID | None — used to scope retrieval queries

# Define structured citation data models
class CitationModel(BaseModel):
    source: str = Field(description="Filename of the source document, e.g. RELIANCE_2025.pdf")
    page: int = Field(description="Page number where the cited text resides")
    quote: str = Field(description="The exact snippet of text cited from the document")

class GroundedAnswer(BaseModel):
    answer: str = Field(description="The synthesized analytical answer, containing inline citations in the format [source, Page X]")
    citations: list[CitationModel] = Field(description="Structured list of all citations used to ground the answer")

# Initialize OpenAI-compatible model pointing to the Groq gateway model requested by the user
provider = OpenAIProvider(
    base_url="https://api.groq.com/openai/v1",
    api_key=settings.GROQ_API_KEY
)

model = OpenAIChatModel(
    model_name=settings.GROQ_LLM_MODEL,  # Dynamically loads "openai/gpt-oss-120b"
    provider=provider
)

# Load the system instructions
instructions_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instructions.md")
system_prompt = ""
try:
    with open(instructions_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
except Exception as e:
    logger.error(f"Failed to read instructions.md system prompt: {e}")
    system_prompt = "You are a grounded Financial Analysis Assistant. Answer questions using only provided context."

# Initialize PydanticAI Agent
agent = Agent(
    model=model,
    deps_type=DocumentAgentDeps,
    system_prompt=system_prompt
)

@agent.system_prompt
def inject_retrieved_context(ctx: RunContext[DocumentAgentDeps]) -> str:
    """
    Injects the top-ranked retrieved document passages directly into the system prompt.
    """
    if not ctx.deps.retrieved_chunks:
        return "No relevant documents found for this query."
        
    formatted = ["Here are the retrieved relevant document passages you must use to ground your answer:\n"]
    for idx, r in enumerate(ctx.deps.retrieved_chunks):
        c = r["chunk"]
        score = r["score"]
        formatted.append(
            f"--- Context Segment {idx + 1} (Score: {score:.4f}) ---\n"
            f"Database Chunk ID (only for tool usage): {c.id}\n"
            f"Source Filename (FILENAME for citation): {c.document.filename}\n"
            f"Page Number: {c.page_number}\n"
            f"Report Section: {c.section_name or 'N/A'}\n"
            f"Text Excerpt:\n{c.text_content}\n"
        )
    return "\n".join(formatted)

@agent.tool
def search_filings(ctx: RunContext[DocumentAgentDeps], query: str) -> str:
    """
    Performs a supplementary hybrid search (semantic + full-text) on the database for a new query
    and returns matching passages with their document source name and page numbers.
    """
    from app.retrieval.retriever import retrieve_hybrid
    logger.info(f"Agent invoked search_filings tool for query: '{query}'")
    try:
        results = retrieve_hybrid(ctx.deps.db, query, user_id=ctx.deps.user_id, limit=4, fetch_neighbors=False)
        if not results:
            return "No matching filings found for this query."
            
        # Register dynamically fetched chunks with context dependencies for grounding validation
        ctx.deps.retrieved_chunks.extend(results)
            
        formatted = []
        for idx, r in enumerate(results):
            c = r["chunk"]
            formatted.append(
                f"Result {idx + 1}:\n"
                f"Database Chunk ID (only for tool usage): {c.id}\n"
                f"Source Filename (FILENAME for citation): {c.document.filename}\n"
                f"Page: {c.page_number}\n"
                f"Content: {c.text_content}\n"
                f"---"
            )
        return "\n\n".join(formatted)
    except Exception as err:
        logger.error(f"Agent tool search_filings failed: {err}")
        return f"Search error occurred: {err}"

@agent.tool
def read_chunk(ctx: RunContext[DocumentAgentDeps], chunk_id: str) -> str:
    """
    Retrieves the full text content of a specific chunk by its chunk ID.
    Useful for reading details of a chunk that was only partially returned.
    """
    logger.info(f"Agent invoked read_chunk tool for ID: {chunk_id}")
    try:
        uuid_id = UUID(chunk_id)
        chunk = ctx.deps.db.query(DocumentChunk).filter(DocumentChunk.id == uuid_id).first()
        if not chunk:
            return f"Chunk ID {chunk_id} not found."
            
        # Register read chunk for grounding verification
        if not any(r["chunk"].id == chunk.id for r in ctx.deps.retrieved_chunks):
            ctx.deps.retrieved_chunks.append({
                "chunk": chunk,
                "score": 1.0,
                "preceding_chunk": None,
                "succeeding_chunk": None
            })
            
        return chunk.text_content
    except Exception as err:
        return f"Error reading chunk: {err}"

@agent.tool
def read_surrounding_chunks(ctx: RunContext[DocumentAgentDeps], chunk_id: str) -> str:
    """
    Retrieves the full text content of the chunks directly preceding and following a specific chunk.
    Use this tool to pad context window and understand surrounding text.
    """
    logger.info(f"Agent invoked read_surrounding_chunks tool for ID: {chunk_id}")
    try:
        uuid_id = UUID(chunk_id)
        db = ctx.deps.db
        chunk = db.query(DocumentChunk).filter(DocumentChunk.id == uuid_id).first()
        if not chunk:
            return f"Chunk ID {chunk_id} not found."
            
        preceding = db.query(DocumentChunk).filter(
            DocumentChunk.source_document_id == chunk.source_document_id,
            DocumentChunk.chunk_index == chunk.chunk_index - 1
        ).first()
        
        succeeding = db.query(DocumentChunk).filter(
            DocumentChunk.source_document_id == chunk.source_document_id,
            DocumentChunk.chunk_index == chunk.chunk_index + 1
        ).first()
        
        # Register read surrounding chunks for grounding verification
        if preceding and not any(r["chunk"].id == preceding.id for r in ctx.deps.retrieved_chunks):
            ctx.deps.retrieved_chunks.append({
                "chunk": preceding,
                "score": 1.0,
                "preceding_chunk": None,
                "succeeding_chunk": None
            })
        if succeeding and not any(r["chunk"].id == succeeding.id for r in ctx.deps.retrieved_chunks):
            ctx.deps.retrieved_chunks.append({
                "chunk": succeeding,
                "score": 1.0,
                "preceding_chunk": None,
                "succeeding_chunk": None
            })
            
        output = []
        if preceding:
            output.append(
                f"=== Preceding Chunk (Index {preceding.chunk_index}, Page {preceding.page_number}) ===\n"
                f"{preceding.text_content}"
            )
        if succeeding:
            output.append(
                f"=== Succeeding Chunk (Index {succeeding.chunk_index}, Page {succeeding.page_number}) ===\n"
                f"{succeeding.text_content}"
            )
            
        return "\n\n".join(output) if output else "No adjacent chunks found."
    except Exception as err:
        return f"Error reading surrounding chunks: {err}"

@agent.tool
def list_available_documents(ctx: RunContext[DocumentAgentDeps]) -> str:
    """
    Lists all source documents currently indexed in the database that are accessible
    to the current analyst (public filings + their own private uploads).
    Use this tool to answer questions like 'which companies can I ask about?'
    or 'what documents do you have access to?'.
    """
    from sqlalchemy import or_
    from app.database.models.source_document import SourceDocument
    logger.info("Agent invoked list_available_documents tool")
    try:
        db = ctx.deps.db
        user_id = ctx.deps.user_id

        query = db.query(SourceDocument)
        if user_id:
            query = query.filter(
                or_(SourceDocument.user_id.is_(None), SourceDocument.user_id == user_id)
            )
        else:
            query = query.filter(SourceDocument.user_id.is_(None))

        docs = query.order_by(SourceDocument.created_at).all()

        if not docs:
            return "No documents are currently indexed in the database."

        lines = ["The following documents are currently available for analysis:\n"]
        for doc in docs:
            scope = "(your private upload)" if doc.user_id else "(public corpus)"
            lines.append(f"- **{doc.filename}** {scope}")
        return "\n".join(lines)
    except Exception as err:
        return f"Error listing documents: {err}"
