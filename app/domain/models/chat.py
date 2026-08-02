"""Chat Domain Entities and Schemas."""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    query: str = Field(..., description="The user prompt or query to be processed by JARVIS Brain.")
    session_id: Optional[str] = Field(default=None, description="Optional session identifier for tracking state.")
    conversation_history: List[ChatMessage] = Field(default_factory=list, description="Prior conversation context.")
    override_provider: Optional[str] = Field(default=None, description="Optional runtime provider override.")
    override_model: Optional[str] = Field(default=None, description="Optional runtime model override.")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
