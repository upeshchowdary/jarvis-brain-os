"""Intent Classification Domain Schemas."""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class IntentCategory(str, Enum):
    GENERAL_CONVERSATION = "general_conversation"
    REASONING_ANALYSIS = "reasoning_analysis"
    TASK_PLANNING = "task_planning"
    CODE_ASSISTANT = "code_assistant"
    TOOL_EXECUTION = "tool_execution"
    KNOWLEDGE_QUERY = "knowledge_query"
    UNKNOWN = "unknown"


class IntentResult(BaseModel):
    category: IntentCategory = Field(default=IntentCategory.GENERAL_CONVERSATION)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    summary: str = Field(..., description="Short summary of user intent")
    extracted_entities: Dict[str, Any] = Field(default_factory=dict)
    suggested_tools: List[str] = Field(default_factory=list)
