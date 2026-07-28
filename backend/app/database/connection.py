from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

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
