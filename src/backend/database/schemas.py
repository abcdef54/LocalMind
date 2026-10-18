from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from pydantic import ConfigDict


from typing import Optional


class ChatRequest(BaseModel):
    conversation_id: UUID
    content: str
    personality: str = "general"
    model_name: str
    reasoning_mode: str = "low"
    temperature: float = 0.0
    tools_enabled: bool = True


class ChatResponse(BaseModel):
    id: UUID
    conversation_id: UUID
    content: str
    role: str
    personality: Optional[str] = None
    created_at: datetime
    citations: list["AttachmentReponse"]

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)


class ConversationCreate(BaseModel):
    user_id: Optional[str] = None
    title: str


class ConversationResponse(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class AttachmentReponse(BaseModel):
    id: UUID
    file_name: str
    file_type: str
    file_size: int
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)
