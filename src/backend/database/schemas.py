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


class InstalledModelResponse(BaseModel):
    id: UUID
    display_name: str
    model_name: str
    hf_repo: str
    gguf_file: str
    local_path: str
    quantization: str
    size_bytes: int
    installed_at: datetime
    is_default: bool
    model_config = ConfigDict(from_attributes=True)


class HFModelBrowseResponse(BaseModel):
    id: str
    author: str
    name: str
    family: str
    size: Optional[str] = None
    task: str
    downloads: int
    downloads_text: str
    likes: int
    license: Optional[str] = None
    created_at: Optional[str] = None


class HFQuantResponse(BaseModel):
    filename: str
    quant: str
    size_bytes: int
    size_gb: float


class DownloadModelRequest(BaseModel):
    repo_id: str
    gguf_filename: str
