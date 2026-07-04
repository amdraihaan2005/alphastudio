from sqlalchemy import Column, String, Integer, ForeignKey, Text, Computed, Index
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
import uuid
from app.database.connection import Base
from app.database.models.constants import EMBEDDING_DIMENSIONS, TEXT_SEARCH_CONFIG

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    __table_args__ = (
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),
        Index(
            "ix_document_chunks_search_vector_gin",
            "search_vector",
            postgresql_using="gin"
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_document_id = Column(UUID(as_uuid=True), ForeignKey("source_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)
    section_name = Column(String, nullable=True)
    text_content = Column(Text, nullable=False)
    
    embedding = Column(Vector(EMBEDDING_DIMENSIONS), nullable=True)
    search_vector = Column(TSVECTOR, Computed(f"to_tsvector('{TEXT_SEARCH_CONFIG}', text_content)", persisted=True))

    document = relationship("SourceDocument", back_populates="chunks")
