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
        task = tracker.current()
        if not task:
            return ToolResult.error("There is no active task to plan.")
        if ctx.task_id is not None and task.id != ctx.task_id:  # another turn started a new task meanwhile
            return ToolResult.error("The task this request belonged to is no longer current; plan not changed.")
        if action in ("set", "revise"):
            if not isinstance(subgoals, list) or not subgoals or not all(isinstance(x, str) and x.strip() for x in subgoals):
                return ToolResult.error("subgoals must be a non-empty list of non-empty strings.")
            tracker.set_plan(subgoals, keep_status=(action == "revise"), task_id=task.id)
        elif action in ("done", "failed"):
            if isinstance(index, bool) or not isinstance(index, int):
                return ToolResult.error("index must be an integer plan position (0-based).")
            if not 0 <= index < len(task.plan):
                return ToolResult.error(f"index {index} is out of range; the plan has {len(task.plan)} item(s).")
            tracker.mark_plan(index, action, task_id=task.id)
        else:
            return ToolResult.error(f"Unknown action '{action}'.")
        return ToolResult.ok(task.summary(max_steps=0))
