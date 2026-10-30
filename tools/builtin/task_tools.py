"""task_plan: lets the model keep an explicit plan for the current multi-step task."""

from __future__ import annotations

from typing import Any, List, Optional

from brain import task_state
from tools.base import Tool, ToolContext, ToolResult, obj
from tools.registry import register


@register
class TaskPlan(Tool):
    name = "task_plan"
    description = (
        "Keep a short plan for the current multi-step task. action='set'/'revise' with subgoals "
        "(ordered list); action='done'/'failed' with the 0-based index of a finished subgoal."
    )
    parameters = obj(
        {
            "action": {"type": "string", "enum": ["set", "done", "failed", "revise"]},
            "subgoals": {"type": "array", "items": {"type": "string"}},
            "index": {"type": "integer", "description": "0-based plan item (for done/failed)"},
        },
        ["action"],
    )
    final = False  # the model continues working after updating the plan

    async def run(self, ctx: ToolContext, action: str = "", subgoals: Optional[List[str]] = None,
                  index: Optional[int] = None, **_: Any) -> ToolResult:
        tracker = task_state.tasks
        if not tracker.current():
            return ToolResult.error("There is no active task to plan.")
        if action in ("set", "revise"):
            if not subgoals:
                return ToolResult.error("Give the subgoals for the plan.")
            tracker.set_plan(subgoals)
        elif action in ("done", "failed"):
            if index is None:
                return ToolResult.error("Which plan item (index)?")
            tracker.mark_plan(int(index), action)
        else:
            return ToolResult.error(f"Unknown action '{action}'.")
        return ToolResult.ok(tracker.current().summary(max_steps=0))
