from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import settings

# Create database engine using the formatted connection URL
# pool_pre_ping checks the connection validity before queries (highly recommended for hosted databases like Supabase)
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True
)

# Session factory for generating database sessions
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Declarative base class for SQLAlchemy models
Base = declarative_base()

def get_db():
    """
    Dependency generator to yield a database session.
    Ensures the session is cleanly closed after the API request finishes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
