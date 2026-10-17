"""
LangChain Document Assistant Agent powered natively by GroqCloud (ChatGroq).

Implements:
- LangChain v1 `create_agent` compiling a state graph for tool-calling agent loops.
- Reusable `ChatGroq` model client with connection pooling.
- LangChain `@tool` definitions using `ToolRuntime[DocumentAgentDeps]` for dependency injection:
  - `search_filings`: Multi-query hybrid retrieval via MultiQueryHybridRetriever.
  - `read_chunk`: Reads full chunk text by ID.
  - `read_surrounding_chunks`: Expands context to adjacent chunk indices.
  - `list_available_documents`: Lists documents indexed for the user.
- `@dynamic_prompt` middleware for runtime context assembly and citation grounding.
- Full `langchain_core.runnables.Runnable` protocol implementation (invoke, ainvoke, stream, astream).
"""

import logging
import os
from typing import Any, Sequence
from uuid import UUID

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from app.config import settings
from app.database.models.document_chunk import DocumentChunk
from app.database.models.source_document import SourceDocument
from app.retrieval.retriever import retrieve_hybrid

logger = logging.getLogger(__name__)


class DocumentAgentDeps:
    """Dependency container carrying database session and retrieved context pool."""

    def __init__(
        self,
        db: Session,
        retrieved_chunks: list[Any],
        user_id: UUID | None = None,
    ):
        self.db = db
        self.retrieved_chunks = retrieved_chunks
        self.user_id = user_id


# Alias for modern LangChain context naming conventions
DocumentAgentContext = DocumentAgentDeps


class AgentRunResult:
    """Encapsulates the agent's output conforming to orchestrator expectations and Runnable conventions."""

    def __init__(
        self,
        output: str,
        messages: list[BaseMessage] | None = None,
        state: dict[str, Any] | None = None,
    ):
        self.output = output
        self.data = output
        self.content = output
        self.messages = messages or []
        self.state = state or {}

    def __str__(self) -> str:
        return self.output

    def __repr__(self) -> str:
        return f"AgentRunResult(output={self.output!r})"

    def __getitem__(self, key: str) -> Any:
        if key in ("output", "data", "content"):
            return self.output
        if key == "messages":
            return self.messages
        if self.state and key in self.state:
            return self.state[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


_instructions_cache: str | None = None


def get_system_instructions() -> str:
    """Loads system instructions from instructions.md with memoization."""
    global _instructions_cache
    if _instructions_cache is None:
        instructions_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "instructions.md"
        )
        try:
            with open(instructions_path, "r", encoding="utf-8") as f:
                _instructions_cache = f.read()
        except Exception as e:
            logger.error(f"Failed to read instructions.md system prompt: {e}")
            _instructions_cache = (
                "You are a grounded Financial Analysis Assistant. Answer questions using only provided context."
            )
    return _instructions_cache


def format_context_segments(chunks: Sequence[Any]) -> str:
    """Formats context chunks (both LangChain Document and model dict) into system prompt excerpts."""
    if not chunks:
        return "No relevant documents found for this query."

    formatted = [
        "Here are the retrieved relevant document passages you must use to ground your answer:\n"
    ]
    for idx, r in enumerate(chunks):
        if hasattr(r, "page_content") and hasattr(r, "metadata"):
            cid = r.metadata.get("chunk_id", "N/A")
            source = r.metadata.get("source", "unknown.pdf")
            page = r.metadata.get("page", "N/A")
            section = r.metadata.get("section_name", "General")
            score = r.metadata.get("score", 0.0)
            text = r.page_content
        elif isinstance(r, dict) and "chunk" in r:
            c = r["chunk"]
            cid = getattr(c, "id", "N/A")
            source = (
                getattr(c.document, "filename", "unknown.pdf")
                if getattr(c, "document", None)
                else "unknown.pdf"
            )
            page = getattr(c, "page_number", "N/A")
            section = getattr(c, "section_name", "General") or "General"
            score = r.get("score", 0.0)
            text = getattr(c, "text_content", "")
        else:
            continue

        formatted.append(
            f"--- Context Segment {idx + 1} (Score: {score:.4f}) ---\n"
            f"Database Chunk ID (only for tool usage): {cid}\n"
            f"Source Filename (FILENAME for citation): {source}\n"
            f"Page Number: {page}\n"
            f"Report Section: {section}\n"
            f"Text Excerpt:\n{text}\n"
        )
    return "\n".join(formatted)


@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    """
    Official LangChain dynamic prompt middleware.
    Injects instructions and the latest context pool dynamically into every model turn.
    """
    deps = request.runtime.context if hasattr(request.runtime, "context") else None
    chunks = deps.retrieved_chunks if deps and hasattr(deps, "retrieved_chunks") else []
    context_text = format_context_segments(chunks)
    return f"{get_system_instructions()}\n\n{context_text}"


@tool
def search_filings(
    query: str,
    runtime: ToolRuntime[DocumentAgentDeps],
    filenames: list[str] | None = None,
) -> str:
    """
    Performs a supplementary hybrid search (semantic + full-text) on the database for a new query
    and returns matching passages with their document source name and page numbers.
    Optionally filter by specific document filenames for multi-file comparisons.
    """
    logger.info(f"Agent invoked search_filings tool for query: '{query}', filenames: {filenames}")
    deps = runtime.context if runtime and hasattr(runtime, "context") else None
    if not deps or not deps.db:
        return "Search error: No database session available in execution context."

    try:
        results = retrieve_hybrid(
            deps.db,
            query,
            user_id=deps.user_id,
            filenames=filenames,
            limit=4,
            fetch_neighbors=False,
        )
        if not results:
            return "No matching filings found for this query."

        deps.retrieved_chunks.extend(results)

        formatted = []
        for idx, r in enumerate(results):
            c = r["chunk"]
            fname = c.document.filename if c.document else "unknown.pdf"
            formatted.append(
                f"Result {idx + 1}:\n"
                f"Database Chunk ID (only for tool usage): {c.id}\n"
                f"Source Filename (FILENAME for citation): {fname}\n"
                f"Page: {c.page_number}\n"
                f"Content: {c.text_content}\n"
                f"---"
            )
        return "\n\n".join(formatted)
    except Exception as err:
        logger.error(f"Agent tool search_filings failed: {err}", exc_info=True)
        return f"Search error occurred: {err}"


@tool
def read_chunk(chunk_id: str, runtime: ToolRuntime[DocumentAgentDeps]) -> str:
    """
    Retrieves the full text content of a specific chunk by its chunk ID.
    Useful for reading details of a chunk that was only partially returned.
    """
    logger.info(f"Agent invoked read_chunk tool for ID: {chunk_id}")
    deps = runtime.context if runtime and hasattr(runtime, "context") else None
    if not deps or not deps.db:
        return "Error: No database session available in execution context."

    try:
        uuid_id = UUID(chunk_id)
        chunk = (
            deps.db.query(DocumentChunk)
            .options(joinedload(DocumentChunk.document))
            .filter(DocumentChunk.id == uuid_id)
            .first()
        )
        if not chunk:
            return f"Chunk ID {chunk_id} not found."

        if (
            chunk.document
            and chunk.document.user_id is not None
            and chunk.document.user_id != deps.user_id
        ):
            return f"Access Denied: You do not have permission to view chunk {chunk_id}."

        # Append to context pool so citing this chunk passes grounding validation
        deps.retrieved_chunks.append(
            {
                "chunk": chunk,
                "score": 1.0,
                "preceding_chunk": None,
                "succeeding_chunk": None,
            }
        )

        return chunk.text_content
    except Exception as err:
        logger.error(f"Agent tool read_chunk failed: {err}", exc_info=True)
        return f"Error reading chunk: {err}"


@tool
def read_surrounding_chunks(chunk_id: str, runtime: ToolRuntime[DocumentAgentDeps]) -> str:
    """
    Retrieves the full text content of the chunks directly preceding and following a specific chunk.
    Use this tool to pad context window and understand surrounding text.
    """
    logger.info(f"Agent invoked read_surrounding_chunks tool for ID: {chunk_id}")
    deps = runtime.context if runtime and hasattr(runtime, "context") else None
    if not deps or not deps.db:
        return "Error: No database session available in execution context."

    try:
        uuid_id = UUID(chunk_id)
        db = deps.db
        chunk = (
            db.query(DocumentChunk)
            .options(joinedload(DocumentChunk.document))
            .filter(DocumentChunk.id == uuid_id)
            .first()
        )
        if not chunk:
            return f"Chunk ID {chunk_id} not found."

        if (
            chunk.document
            and chunk.document.user_id is not None
            and chunk.document.user_id != deps.user_id
        ):
            return f"Access Denied: You do not have permission to view chunk {chunk_id}."

        preceding = (
            db.query(DocumentChunk)
            .options(joinedload(DocumentChunk.document))
            .filter(
                DocumentChunk.source_document_id == chunk.source_document_id,
                DocumentChunk.chunk_index == chunk.chunk_index - 1,
            )
            .first()
        )

        succeeding = (
            db.query(DocumentChunk)
            .options(joinedload(DocumentChunk.document))
            .filter(
                DocumentChunk.source_document_id == chunk.source_document_id,
                DocumentChunk.chunk_index == chunk.chunk_index + 1,
            )
            .first()
        )

        if preceding:
            deps.retrieved_chunks.append(
                {
                    "chunk": preceding,
                    "score": 1.0,
                    "preceding_chunk": None,
                    "succeeding_chunk": None,
                }
            )
        if succeeding:
            deps.retrieved_chunks.append(
                {
                    "chunk": succeeding,
                    "score": 1.0,
                    "preceding_chunk": None,
                    "succeeding_chunk": None,
                }
            )

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
        logger.error(f"Agent tool read_surrounding_chunks failed: {err}", exc_info=True)
        return f"Error reading surrounding chunks: {err}"


@tool
def list_available_documents(runtime: ToolRuntime[DocumentAgentDeps]) -> str:
    """
    Lists all source documents currently indexed in the database that are accessible
    to the current analyst (their private uploads).
    Use this tool to answer questions like 'which documents do you have access to?'.
    """
    logger.info("Agent invoked list_available_documents tool")
    deps = runtime.context if runtime and hasattr(runtime, "context") else None
    if not deps or not deps.db:
        return "Error: No database session available in execution context."

    try:
        db = deps.db
        user_id = deps.user_id

        query = db.query(SourceDocument)
        if user_id:
            query = query.filter(
                or_(SourceDocument.user_id.is_(None), SourceDocument.user_id == user_id)
            )
        else:
            query = query.filter(SourceDocument.user_id.is_(None))

        docs = query.order_by(SourceDocument.created_at).all()
        if not docs:
            return "No documents found in the database for your account."

        lines = ["Indexed Documents:"]
        for d in docs:
            lines.append(f"- {d.filename} (Uploaded: {d.created_at})")
        return "\n".join(lines)
    except Exception as err:
        logger.error(f"Agent tool list_available_documents failed: {err}", exc_info=True)
        return f"Error listing documents: {err}"


AGENT_TOOLS: list[BaseTool] = [
    search_filings,
    read_chunk,
    read_surrounding_chunks,
    list_available_documents,
]


def make_agent_tools(deps: DocumentAgentDeps | None = None) -> list[BaseTool]:
    """Returns canonical LangChain agent tools for document analysis."""
    return AGENT_TOOLS


class LangChainDocumentAgent(Runnable[dict[str, Any] | str, AgentRunResult]):
    """
    Official LangChain Document Agent powered by GroqCloud (ChatGroq).

    Conforms natively to the LangChain `Runnable` interface:
    - Built via LangChain v1 `create_agent` factory compiling a state graph.
    - Uses `@dynamic_prompt` middleware for runtime grounding prompt composition.
    - Uses `ToolRuntime[DocumentAgentDeps]` for type-safe dependency injection without prompt pollution.
    - Provides high-performance, single-instance client reuse with async connection pooling.
    """

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float = 0.0,
        llm: Any = None,
        tools: Sequence[BaseTool] | None = None,
    ):
        self.model_name = model_name or settings.GROQ_LLM_MODEL
        self.temperature = temperature
        self._custom_llm = llm
        self.tools = list(tools) if tools is not None else list(AGENT_TOOLS)
        self.tool_names = [t.name for t in self.tools]
        self._cached_llm: Any = None
        self._cached_graph: Any = None

    def _get_llm(self) -> Any:
        if self._custom_llm is not None:
            return self._custom_llm
        return ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=self.model_name,
            temperature=self.temperature,
        )

    def get_graph(self) -> Any:
        """Retrieves or compiles the underlying LangGraph CompiledStateGraph."""
        llm = self._get_llm()
        # When mocked (e.g. in unit tests with patch), compile graph with mock
        if type(llm).__name__ in ("MagicMock", "AsyncMock") or hasattr(
            llm, "_mock_return_value"
        ):
            return create_agent(
                model=llm,
                tools=self.tools,
                middleware=[dynamic_system_prompt],
                context_schema=DocumentAgentDeps,
            )

        if self._cached_graph is None or self._cached_llm is not llm:
            self._cached_llm = llm
            self._cached_graph = create_agent(
                model=llm,
                tools=self.tools,
                middleware=[dynamic_system_prompt],
                context_schema=DocumentAgentDeps,
            )
        return self._cached_graph

    async def ainvoke(
        self,
        input: dict[str, Any] | str | list[Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AgentRunResult:
        """Asynchronously executes the agent state graph conforming to LangChain Runnable."""
        graph = self.get_graph()
        context = kwargs.get("context") or kwargs.get("deps")

        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        elif isinstance(input, dict) and "messages" in input:
            messages = input["messages"]
        elif isinstance(input, (list, tuple)):
            messages = list(input)
        else:
            messages = [HumanMessage(content=str(input))]

        state = await graph.ainvoke(
            {"messages": messages},
            context=context,
            config=config,
        )

        final_messages = state.get("messages", [])
        last_message = final_messages[-1] if final_messages else AIMessage(content="")
        output_text = str(getattr(last_message, "content", ""))
        return AgentRunResult(output=output_text, messages=final_messages, state=state)

    def invoke(
        self,
        input: dict[str, Any] | str | list[Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AgentRunResult:
        """Synchronously executes the agent state graph conforming to LangChain Runnable."""
        graph = self.get_graph()
        context = kwargs.get("context") or kwargs.get("deps")

        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        elif isinstance(input, dict) and "messages" in input:
            messages = input["messages"]
        elif isinstance(input, (list, tuple)):
            messages = list(input)
        else:
            messages = [HumanMessage(content=str(input))]

        state = graph.invoke(
            {"messages": messages},
            context=context,
            config=config,
        )

        final_messages = state.get("messages", [])
        last_message = final_messages[-1] if final_messages else AIMessage(content="")
        output_text = str(getattr(last_message, "content", ""))
        return AgentRunResult(output=output_text, messages=final_messages, state=state)

    async def astream(
        self,
        input: dict[str, Any] | str | list[Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ):
        """Streams graph update events asynchronously."""
        graph = self.get_graph()
        context = kwargs.get("context") or kwargs.get("deps")
        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        elif isinstance(input, dict) and "messages" in input:
            messages = input["messages"]
        elif isinstance(input, (list, tuple)):
            messages = list(input)
        else:
            messages = [HumanMessage(content=str(input))]

        async for chunk in graph.astream(
            {"messages": messages},
            context=context,
            config=config,
            **kwargs,
        ):
            yield chunk

    def stream(
        self,
        input: dict[str, Any] | str | list[Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ):
        """Streams graph update events synchronously."""
        graph = self.get_graph()
        context = kwargs.get("context") or kwargs.get("deps")
        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        elif isinstance(input, dict) and "messages" in input:
            messages = input["messages"]
        elif isinstance(input, (list, tuple)):
            messages = list(input)
        else:
            messages = [HumanMessage(content=str(input))]

        for chunk in graph.stream(
            {"messages": messages},
            context=context,
            config=config,
            **kwargs,
        ):
            yield chunk

    async def run(self, user_query: str, deps: DocumentAgentDeps) -> AgentRunResult:
        """High-level entry point called by the chat orchestrator."""
        return await self.ainvoke(user_query, context=deps)


# Singleton instance matching product import interface: `from app.assistant.agent import agent`
agent = LangChainDocumentAgent()
