from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.auth.dependencies import get_current_user
from app.database.models.user import User
from app.routers import chat
from app.logging_config import configure_logging

configure_logging()


app = FastAPI(
    title="Alpha Copilot",
    description="FastAPI service for financial document analysis and private RAG using Groq & Cohere",
    version="0.1.0"
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
    Health check endpoint to verify the API service is up and running.
    """
    return {
        "status": "healthy",
        "llm_model": settings.GROQ_LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_dimensions": settings.EMBEDDING_DIMENSIONS
    }

@app.get("/auth/test-me", tags=["Authentication"])
async def test_auth(current_user: User = Depends(get_current_user)):
    """
    Protected test endpoint to verify authorization headers and token validation.
    """
    return {
        "status": "authenticated",
        "user_id": str(current_user.id),
        "email": current_user.email
    }
