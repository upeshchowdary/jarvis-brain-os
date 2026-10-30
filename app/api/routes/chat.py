"""Chat endpoints: request/response and Server-Sent-Events streaming."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from brain.pipeline import TurnResult, brain

router = APIRouter(tags=["Chat"])


class BrainChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000, description="User message")
    session_id: Optional[str] = Field(default=None, max_length=100)
    personality: Optional[str] = Field(default=None, description="assistant | professional | friendly | developer | minimal")
    model_override: Optional[str] = Field(default=None, description="provider/model for this request")
    dry_run: Optional[bool] = Field(default=None, description="Simulate actions instead of executing them")
    channel: str = Field(default="text", pattern="^(text|voice|api)$")


@router.post("/chat")
async def process_chat_query(request: BrainChatRequest) -> Dict[str, Any]:
    result: TurnResult = await brain.handle(
        request.query,
        request.session_id,
        channel=request.channel,
        personality=request.personality,
        model_override=request.model_override,
        dry_run=request.dry_run,
    )
    return result.to_dict()


def _sse(event: Dict[str, Any]) -> str:
    if event["type"] == "done":
        event = {"type": "done", "result": event["result"].to_dict()}
    return f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"


@router.post("/chat/stream")
async def stream_chat(request: BrainChatRequest) -> StreamingResponse:
    """Stream tokens as they are generated (SSE). Events: text, tool, done."""

    async def gen() -> AsyncIterator[str]:
        try:
            async for event in brain.stream(
                request.query,
                request.session_id,
                channel=request.channel,
                personality=request.personality,
                model_override=request.model_override,
                dry_run=request.dry_run,
            ):
                yield _sse(event)
        except Exception as exc:  # surface errors to the client instead of a dropped connection
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
