"""Structured Brain Response Domain Models."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.domain.models.intent import IntentResult
from app.domain.models.plan import ExecutionPlan


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponseMetadata(BaseModel):
    provider: str
    model: str
    latency_ms: float
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: Optional[str] = None


class BrainResponse(BaseModel):
    intent: str = Field(..., description="Detected user query intent category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Intent classification confidence score")
    reasoning_steps: List[str] = Field(default_factory=list, description="Step-by-step reasoning breakdown")
    plan: List[Dict[str, Any]] = Field(default_factory=list, description="Execution plan sub-tasks")
    required_tools: List[str] = Field(default_factory=list, description="Tools required for task completion")
    response: str = Field(..., description="Final response generated for the user")
    metadata: Optional[LLMResponseMetadata] = Field(default=None, description="LLM execution metadata")
