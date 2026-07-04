import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

class Settings(BaseSettings):
    # Supabase configurations
    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    SUPABASE_SERVICE_ROLE_KEY: str
    
    # Database configuration
    DATABASE_URL: str
    
    # LLM configurations
    GROQ_API_KEY: str
    GROQ_LLM_MODEL: str = Field(default="openai/gpt-oss-120b")
    
    # Embedding configurations
    COHERE_API_KEY: str
    EMBEDDING_MODEL: str = Field(default="embed-english-v3.0")
    EMBEDDING_DIMENSIONS: int = Field(default=1024)

    # Allow loading settings from .env file
    # We specify the env file relative to the project root
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @field_validator("DATABASE_URL")
    @classmethod
    def convert_database_url_driver(cls, v: str) -> str:
        # If the URL starts with postgresql://, modify it to use psycopg (Psycopg 3)
        if v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+psycopg://", 1)
        return v

# Global settings instance
settings = Settings()
