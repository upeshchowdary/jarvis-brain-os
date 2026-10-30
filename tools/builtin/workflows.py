"""Workflow tools: record, stop, list and replay (replays go through the gateway)."""

from __future__ import annotations

import re
from typing import Any, Dict

from automation.workflow_player import workflow_player
from automation.workflow_recorder import workflow_recorder
from tools.base import Tool, ToolContext, ToolResult, from_gateway, obj
from tools.registry import register


@register
class StartRecording(Tool):
    name = "workflow_record_start"
    description = "Start recording a reusable workflow with the given name."
    parameters = obj({"name": {"type": "string"}}, ["name"])
    side_effect = True

    async def run(self, ctx: ToolContext, name: str = "my workflow", **_: Any) -> ToolResult:
        res = workflow_recorder.start_recording(name.strip() or "my workflow")
        if not res.get("success"):
            return ToolResult.error(res.get("error", "Couldn't start recording."))
        return ToolResult.ok(f"🔴 Recording workflow “{name}”. Tell me the steps or use the dashboard; say “stop recording” when done.")


@register
class StopRecording(Tool):
    name = "workflow_record_stop"
    description = "Stop and save the workflow that is being recorded."
    parameters = obj({})
    side_effect = True

    async def run(self, ctx: ToolContext, **_: Any) -> ToolResult:
        if not workflow_recorder.is_recording:
            return ToolResult.error("I'm not recording anything right now.")
        stop = workflow_recorder.stop_recording()
        saved = workflow_recorder.save_workflow()
        return ToolResult.ok(f"Saved workflow “{saved.get('name')}” with {stop.get('steps', 0)} steps. Say “replay {saved.get('name')}” to run it.")


@register
class ListWorkflows(Tool):
    name = "workflow_list"
    description = "List saved workflows."
    parameters = obj({})

    async def run(self, ctx: ToolContext, **_: Any) -> ToolResult:
        names = workflow_recorder.list_workflows()
        return ToolResult.ok("Saved workflows: " + ", ".join(names) if names else "No saved workflows yet.", data=names)


def parse_params(text: str) -> Dict[str, str]:
    """'name=Rahul, day=friday' or 'name Rahul' → dict."""
    params: Dict[str, str] = {}
    for key, val in re.findall(r"([a-zA-Z_]\w*)\s*[=:]\s*([^,;]+)", text or ""):
        params[key.strip()] = val.strip()
    return params


@register
class ReplayWorkflow(Tool):
    name = "workflow_replay"
    description = "Replay a saved workflow by name, optionally with parameters (e.g. {\"name\": \"Rahul\"})."
    parameters = obj({"name": {"type": "string"}, "params": {"type": "object", "additionalProperties": {"type": "string"}}}, ["name"])
    side_effect = True

    async def run(self, ctx: ToolContext, name: str = "", params: Any = None, **_: Any) -> ToolResult:
        p = params if isinstance(params, dict) else parse_params(str(params or ""))
        res = await workflow_player.replay(name.strip(), p, ctx.session_id, dry_run=ctx.dry_run, source=ctx.source)
        return from_gateway(res)
