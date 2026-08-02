"""Planner Engine for hierarchical multi-step task breakdown."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class PlanTaskStep(BaseModel):
    step_number: int = Field(..., ge=1)
    action: str = Field(..., description="Short action key for this step")
    description: str = Field(..., description="Detailed description of what is accomplished")
    sub_tasks: List[str] = Field(default_factory=list, description="Hierarchical sub-tasks")
    tool_required: Optional[str] = Field(default=None)


class HierarchicalPlan(BaseModel):
    goal: str
    complexity: str = Field(default="medium")
    steps: List[PlanTaskStep] = Field(default_factory=list)


class Planner:
    """Generates structured hierarchical plans for complex multi-stage tasks."""

    @staticmethod
    def create_plan(goal: str, intent_code: str = "GENERAL") -> HierarchicalPlan:
        """Construct hierarchical plan based on goal and intent."""
        g_lower = goal.lower()

        # Build Website Example
        if "website" in g_lower or "web app" in g_lower:
            steps = [
                PlanTaskStep(
                    step_number=1,
                    action="frontend_design",
                    description="Design dynamic UI and responsive layout",
                    sub_tasks=["Component layout", "CSS theme tokens", "State hooks"],
                ),
                PlanTaskStep(
                    step_number=2,
                    action="backend_api",
                    description="Implement REST API framework and routing controllers",
                    sub_tasks=["Endpoint schema validation", "Controller business logic"],
                ),
                PlanTaskStep(
                    step_number=3,
                    action="database_setup",
                    description="Setup persistence schema and ORM models",
                    sub_tasks=["Tables migration", "Async session pool"],
                ),
                PlanTaskStep(
                    step_number=4,
                    action="authentication",
                    description="Implement JWT authentication and security headers",
                    sub_tasks=["Token issuance", "Password hashing"],
                ),
                PlanTaskStep(
                    step_number=5,
                    action="testing_deployment",
                    description="Run unit test suite and deploy container build",
                    sub_tasks=["Pytest suite", "Production server launch"],
                ),
            ]
            return HierarchicalPlan(goal=goal, complexity="high", steps=steps)

        # Standard Step Plan
        default_steps = [
            PlanTaskStep(
                step_number=1,
                action="analyze_requirements",
                description=f"Deconstruct target goal: '{goal[:60]}...'",
                sub_tasks=["Parse input arguments", "Check environmental parameters"],
            ),
            PlanTaskStep(
                step_number=2,
                action="execute_solution",
                description="Synthesize comprehensive response using LLM reasoning",
                sub_tasks=["Format output payload", "Verify accuracy"],
            ),
        ]
        return HierarchicalPlan(goal=goal, complexity="low", steps=default_steps)


# Global planner singleton
planner = Planner()
