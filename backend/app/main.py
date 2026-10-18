"""
Alpha Copilot FastAPI Application: LangChain RAG & Agentic Document Analysis.

Orchestrates:
- LangChain GroqCloud LLM (`ChatGroq` with `openai/gpt-oss-120b`).
- LangChain Cohere Embeddings (`AlphaCohereEmbeddings` with `embed-english-v3.0`).
- LangChain Multi-Query Hybrid Retriever (`MultiQueryHybridRetriever`).
- LangChain Document Agent (`LangChainDocumentAgent`).
"""

from contextlib import asynccontextmanager
import logging
import sys

# Silence harmless CPython Windows Proactor shutdown cleanup errors on exit/reload
if sys.platform == "win32":
    from functools import wraps
    from asyncio.proactor_events import _ProactorBasePipeTransport

    _orig_del = _ProactorBasePipeTransport.__del__

    @wraps(_orig_del)
    def _silent_del(self, *args, **kwargs):
        try:
            _orig_del(self, *args, **kwargs)
        except RuntimeError as e:
            if str(e) != "Event loop is closed":
                raise

    _ProactorBasePipeTransport.__del__ = _silent_del

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents.transformers import BaseDocumentTransformer
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable

from app.config import settings
from app.auth.dependencies import get_current_user
from app.database.models.user import User
from app.routers import chat
from app.logging_config import configure_logging
from app.assistant.agent import agent
from app.ingest.chunker import RecursiveLangChainChunker
from app.ingest.embedder import get_embeddings_model
from app.ingest.orchestrator import default_orchestrator
from app.retrieval.decomposer import MultiQueryDecomposer, get_default_decomposer, get_groq_llm
from app.retrieval.fusion import ReciprocalRankFusion
from app.retrieval.retriever import MultiQueryHybridRetriever

configure_logging()
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager:
    Verifies that the LangChain AI providers (GroqCloud LLM and Cohere Embeddings)
    are properly configured at startup, binds shared instances to app.state,
    and conducts startup configuration integrity checks.
    """
    logger.info("Initializing Alpha Copilot LangChain Pipeline...")

    # 1. Validate and warm up GroqCloud LLM integration
    app.state.llm = None
    if not settings.GROQ_API_KEY or not settings.GROQ_API_KEY.strip():
        logger.warning(
            "GROQ_API_KEY is not configured. LLM calls and agent execution will fail until configured."
        )
    else:
        logger.info(
            f"LangChain LLM initialized: GroqCloud ChatGroq (model: {settings.GROQ_LLM_MODEL})"
        )
        try:
            app.state.llm = get_groq_llm()
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM on startup: {e}")

    # 2. Validate and warm up Cohere Embeddings integration
    app.state.embeddings = None
    if not settings.COHERE_API_KEY or not settings.COHERE_API_KEY.strip():
        logger.warning(
            "COHERE_API_KEY is not configured. Embeddings and semantic retrieval will fail until configured."
        )
    else:
        logger.info(
            f"LangChain Embeddings initialized: Cohere AlphaCohereEmbeddings (model: {settings.EMBEDDING_MODEL})"
        )
        try:
            app.state.embeddings = get_embeddings_model()
        except Exception as e:
            logger.error(f"Failed to initialize Cohere Embeddings on startup: {e}")

    # 3. Mount canonical LangChain Document Transformer
    app.state.chunker = RecursiveLangChainChunker()
    logger.info(
        f"LangChain Document Transformer mounted: {RecursiveLangChainChunker.__name__} "
        f"(BaseDocumentTransformer: {issubclass(RecursiveLangChainChunker, BaseDocumentTransformer)}, "
        f"Runnable: {issubclass(RecursiveLangChainChunker, Runnable)})"
    )

    # 4. Mount canonical LangChain Ingestion Orchestrator
    app.state.orchestrator = default_orchestrator
    logger.info(
        f"LangChain Ingest Orchestrator mounted: {default_orchestrator.__class__.__name__} "
        f"(Runnable: {isinstance(default_orchestrator, Runnable)})"
    )

    # 5. Mount canonical LangChain Document Agent
    app.state.agent = agent
    logger.info(
        f"LangChain Agent mounted: {agent.__class__.__name__} (Runnable: {isinstance(agent, Runnable)}, Tools: {agent.tool_names})"
    )

    # 6. Mount canonical LangChain BaseRetriever Class
    app.state.retriever_class = MultiQueryHybridRetriever
    logger.info(
        f"LangChain Retriever mounted: {MultiQueryHybridRetriever.__name__} "
        f"(BaseRetriever: {issubclass(MultiQueryHybridRetriever, BaseRetriever)}, "
        f"Runnable: {issubclass(MultiQueryHybridRetriever, Runnable)})"
    )

    # 7. Mount canonical LangChain Multi-Query Decomposer
    app.state.decomposer = get_default_decomposer()
    logger.info(
        f"LangChain Decomposer mounted: {MultiQueryDecomposer.__name__} "
        f"(Runnable: {isinstance(app.state.decomposer, Runnable)})"
    )

    # 8. Mount canonical LangChain RRF Fusion
    app.state.fusion = ReciprocalRankFusion()
    logger.info(
        f"LangChain Fusion mounted: {ReciprocalRankFusion.__name__} "
        f"(Runnable: {isinstance(app.state.fusion, Runnable)})"
    )

    yield

    logger.info("Shutting down Alpha Copilot LangChain services.")


app = FastAPI(
    title="Alpha Copilot",
    description="LangChain-powered service for financial document analysis and private RAG using Groq (openai/gpt-oss-120b) & Cohere (embed-english-v3.0)",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint returning active LangChain pipeline and model specifications.
    Dynamically introspects registered LangChain tools and framework capabilities.
    """
    import langchain
    import langchain_core

    return {
        "status": "healthy",
        "framework": "LangChain",
        "versions": {
            "langchain": getattr(langchain, "__version__", "unknown"),
            "langchain_core": getattr(langchain_core, "__version__", "unknown"),
        },
        "llm": {
            "provider": "GroqCloud",
            "class": "ChatGroq",
            "model": settings.GROQ_LLM_MODEL,
        },
        "embeddings": {
            "provider": "Cohere",
            "class": "AlphaCohereEmbeddings",
            "model": settings.EMBEDDING_MODEL,
            "dimensions": settings.EMBEDDING_DIMENSIONS,
        },
        "retrieval": {
            "class": "MultiQueryHybridRetriever",
            "fusion": "Reciprocal Rank Fusion (RRF)",
        },
        "agent": {
            "class": agent.__class__.__name__,
            "is_runnable": isinstance(agent, Runnable),
            "tools": [t.name for t in agent.tools],
        },
    }


@app.get("/langchain/pipeline", tags=["LangChain Diagnostics"])
async def pipeline_info():
    """
    Detailed LangChain architecture summary for client diagnostics and introspection.
    Validates genuine inheritance and interface conformance across all pipeline components.
    """
    return {
        "pipeline": "Alpha Copilot Agentic RAG",
        "components": {
            "document_loader": "PyPDFToLangChainParser (inherits BaseBlobParser, emits langchain_core.documents.Document)",
            "document_transformer": "RecursiveLangChainChunker (inherits BaseDocumentTransformer)",
            "embeddings": "AlphaCohereEmbeddings (inherits langchain_core.embeddings.Embeddings)",
            "orchestrator": "LangChainIngestOrchestrator (inherits langchain_core.runnables.Runnable, pipelined sync-to-async stream)",
            "decomposer": f"{MultiQueryDecomposer.__name__} (inherits langchain_core.runnables.RunnableSerializable)",
            "fusion": f"{ReciprocalRankFusion.__name__} (inherits langchain_core.runnables.RunnableSerializable, weights & Document deduplication)",
            "retriever": f"{MultiQueryHybridRetriever.__name__} (inherits langchain_core.retrievers.BaseRetriever, Runnable: {issubclass(MultiQueryHybridRetriever, Runnable)})",
            "agent": f"{agent.__class__.__name__} (inherits langchain_core.runnables.Runnable, compiled via langchain.agents.create_agent)",
        },
        "agent_tools": [
            {
                "name": t.name,
                "description": t.description.strip().split("\n")[0]
                if t.description
                else "",
            }
            for t in agent.tools
        ],
    }


@app.get("/auth/test-me", tags=["Authentication"])
async def test_auth(current_user: User = Depends(get_current_user)):
    """
    Protected test endpoint to verify authorization headers and token validation.
    """
    return {
        "status": "authenticated",
        "user_id": str(current_user.id),
        "email": current_user.email,
    }
