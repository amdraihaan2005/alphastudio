import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    SUPABASE_SERVICE_ROLE_KEY: str

    DATABASE_URL: str

    GROQ_API_KEY: str
    GROQ_LLM_MODEL: str = Field(default="openai/gpt-oss-120b")

    COHERE_API_KEY: str
    EMBEDDING_MODEL: str = Field(default="embed-english-v3.0")
    EMBEDDING_DIMENSIONS: int = Field(default=1024)

    GET_EMBEDDINGS_RETRIES: int = Field(default=5)
    GET_EMBEDDINGS_DELAY: float = Field(default=5.0)
    GET_QUERY_RETRIES: int = Field(default=4)
    GET_QUERY_DELAY: float = Field(default=2.0)
    COHERE_MAX_BATCH_SIZE: int = Field(default=96)
    MAX_CONCURRENT_EMBED_REQUESTS: int = Field(default=5)

    model_config = SettingsConfigDict(
        env_file=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("DATABASE_URL")
    @classmethod
    def convert_database_url_driver(cls, v: str) -> str:
        if v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+psycopg://", 1)
        return v


settings = Settings()
