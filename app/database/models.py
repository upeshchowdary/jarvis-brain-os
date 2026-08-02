"""Database Models and Table Definitions for SQLite persistence."""

from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ChatLogRecord(BaseModel):
    """Data record schema for persisting chat execution telemetry in SQLite."""

    id: Optional[int] = None
    session_id: Optional[str] = None
    user_query: str
    intent: str
    confidence: float
    provider: str
    model: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    response_text: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
