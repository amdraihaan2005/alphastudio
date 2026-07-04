from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID
from typing import List

class ThreadBase(BaseModel):
    title: str = Field(default="New Conversation")

class ThreadCreate(ThreadBase):
    pass

class ThreadResponse(ThreadBase):
    id: UUID
    user_id: UUID
    created_at: datetime

    model_config = {
        "from_attributes": True
    }

class MessageCitationSchema(BaseModel):
    chunk_id: UUID
    filename: str
    page_number: int

    model_config = {
        "from_attributes": True
    }

class MessageResponse(BaseModel):
    id: UUID
    chat_thread_id: UUID
    role: str
    content: str
    created_at: datetime
    citations: List[MessageCitationSchema] = []

    model_config = {
        "from_attributes": True
    }

class ChatMessageCreate(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatStreamRequest(BaseModel):
    thread_id: UUID
    messages: List[ChatMessageCreate]
