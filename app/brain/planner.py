"""Planning Engine for multi-step execution plan generation."""

from typing import List, Dict, Any
from app.domain.models.plan import ExecutionPlan, PlanStep


class Planner:
    """Builds and validates step-by-step execution plans."""

    @staticmethod
    def parse_plan(raw_plan: Any, query: str) -> ExecutionPlan:
        """Parse raw JSON plan into ExecutionPlan entity."""
        steps = []
        if isinstance(raw_plan, list):
            for idx, item in enumerate(raw_plan, start=1):
                if isinstance(item, dict):
                    step = PlanStep(
                        step_number=item.get("step_number", idx),
                        action=item.get("action", f"step_{idx}"),
                        description=item.get("description", str(item)),
                        tool_name=item.get("tool_name"),
                        parameters=item.get("parameters", {}),
                    )
                    steps.append(step)
                elif isinstance(item, str):
                    step = PlanStep(
                        step_number=idx,
                        action=f"step_{idx}",
                        description=item,
                    )
                    steps.append(step)

        if not steps:
            steps.append(
                PlanStep(
                    step_number=1,
                    action="direct_response",
                    description=f"Direct answer generated for query: '{query[:50]}...'",
                )
            )

        return ExecutionPlan(
            goal=query,
            steps=steps,
            estimated_complexity="low" if len(steps) <= 2 else "medium",
        )
