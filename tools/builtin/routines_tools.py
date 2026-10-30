"""Routine tools: run / list YAML routines (modes)."""

from __future__ import annotations

from typing import Any

from services import routines
from tools.base import Tool, ToolContext, ToolResult, from_gateway, obj
from tools.registry import register


@register
class RunRoutine(Tool):
    name = "run_routine"
    description = "Run a saved routine/mode (e.g. 'study mode', 'shutdown routine', 'morning'), optionally with variables."
    parameters = obj({"name": {"type": "string"}, "variables": {"type": "object", "additionalProperties": {"type": "string"}}}, ["name"])
    side_effect = True

    async def run(self, ctx: ToolContext, name: str = "", variables: Any = None, **_: Any) -> ToolResult:
        result, messages = await routines.run(name, ctx.session_id, dict(variables or {}), dry_run=ctx.dry_run, source=ctx.source)
        if result is not None and result.status in ("needs_confirmation", "failed", "blocked"):
            return from_gateway(result)
        if result is None and messages and messages[0].startswith("I don't have"):
            return ToolResult.error(messages[0])
        status = "dry_run" if result is not None and result.status == "dry_run" else "ok"
        tr = ToolResult.ok("\n".join(m for m in messages if m))
        tr.status = status
        return tr


@register
class ListRoutines(Tool):
    name = "list_routines"
    description = "List available routines/modes."
    parameters = obj({})

    async def run(self, ctx: ToolContext, **_: Any) -> ToolResult:
        all_routines = routines.load_all()
        if not all_routines:
            return ToolResult.ok("No routines yet. Add YAML files to data/routines/.")
        lines = [f"• {name} — {r.get('description', '')}" for name, r in all_routines.items()]
        return ToolResult.ok("Routines:\n" + "\n".join(lines), data=list(all_routines))
