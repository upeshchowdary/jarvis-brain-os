"""Chat Processing REST Endpoint delegating to BrainManager Framework."""

from fastapi import APIRouter, Depends
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from brain.brain_manager import brain_manager, BrainExecutionOutput
from app.database.connection import db_manager, DatabaseManager
from app.database.models import ChatLogRecord

router = APIRouter(tags=["Chat"])


class BrainChatRequest(BaseModel):
    query: str = Field(..., description="User query prompt")
    session_id: Optional[str] = Field(default=None)
    personality: Optional[str] = Field(default=None, description="Optional personality override (assistant, developer, etc.)")
    model_override: Optional[str] = Field(default=None, description="Optional model override")


@router.post("/chat")
async def process_chat_query(request: BrainChatRequest) -> Dict[str, Any]:
    """Execute query via JARVIS Brain Framework and return structured execution output."""
    output: BrainExecutionOutput = await brain_manager.execute_cognitive_pipeline(
        user_query=request.query,
        session_id=request.session_id,
        personality=request.personality,
        model_override=request.model_override,
    )

    # Persist execution telemetry to SQLite
    record = ChatLogRecord(
        session_id=output.metadata.get("session_id"),
        user_query=output.query,
        intent=output.intent.intent,
        confidence=output.intent.confidence,
        provider=output.metadata.get("provider", "groq"),
        model=output.metadata.get("model", "llama-3.3-70b-versatile"),
        latency_ms=output.metadata.get("total_latency_ms", 0.0),
        prompt_tokens=output.metadata.get("usage", {}).get("prompt_tokens", 0),
        completion_tokens=output.metadata.get("usage", {}).get("completion_tokens", 0),
        total_tokens=output.metadata.get("usage", {}).get("total_tokens", 0),
        response_text=output.response,
    )
    await db_manager.log_chat_telemetry(record)

    return output.model_dump()
