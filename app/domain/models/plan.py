"""Plan Domain Schemas for JARVIS Multi-Step Reasoning."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    step_number: int = Field(..., ge=1)
    action: str = Field(..., description="Action name or intent for this step")
    description: str = Field(..., description="Human-readable detail of what this step accomplishes")
    tool_name: Optional[str] = Field(default=None, description="Target tool if execution is required")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Arguments for tool execution")
    completed: bool = Field(default=False)


class ExecutionPlan(BaseModel):
    goal: str = Field(..., description="Overall goal of the execution plan")
    steps: List[PlanStep] = Field(default_factory=list)
    estimated_complexity: str = Field(default="low", description="low, medium, or high")
