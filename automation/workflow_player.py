"""Workflow replay — recorded workflows become a ``computer.run_plan`` gateway request.

Parameters: step text may contain ``{name}`` placeholders, filled from ``params``
("replay send report to {name}" → ``params={"name": "Rahul"}``). Replays get the
same preview, confirmation, dry-run, verification/retry and audit as any other plan.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from automation.gateway import ActionRequest, GatewayResult, gateway
from automation.workflow_recorder import workflow_recorder

_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def fill_params(text: str, params: Dict[str, str]) -> str:
    return _PLACEHOLDER.sub(lambda m: str(params.get(m.group(1), m.group(0))), text or "")


def missing_params(steps: List[Dict[str, Any]], params: Dict[str, str]) -> List[str]:
    names = set()
    for s in steps:
        for field in ("target", "value"):
            names.update(_PLACEHOLDER.findall(str(s.get(field, ""))))
    return sorted(n for n in names if n not in params)


class WorkflowPlayer:
    def steps_for(self, name: str, params: Optional[Dict[str, str]] = None) -> Optional[List[Dict[str, Any]]]:
        wf = workflow_recorder.load_workflow(name)
        if wf is None:
            return None
        params = params or {}
        steps = []
        for st in wf.steps:
            step: Dict[str, Any] = {
                "action": {"left_click": "click", "type_text": "type", "press_key": "press", "fill_field": "fill",
                           "switch_window": "focus_window"}.get(st.action, st.action),
                "target": fill_params(st.target, params),
                "value": fill_params(st.value, params),
                "description": fill_params(st.description, params),
            }
            if st.x is not None and st.y is not None and not st.target:
                step["x"], step["y"] = st.x, st.y
            steps.append(step)
        return steps

    def build_request(self, name: str, params: Optional[Dict[str, str]] = None) -> Optional[ActionRequest]:
        steps = self.steps_for(name, params)
        if steps is None:
            return None
        return ActionRequest("computer.run_plan", {"goal": f"Replay workflow '{name}'", "steps": steps}, description=f"Replay '{name}'")

    async def replay(self, name: str, params: Optional[Dict[str, str]] = None, session_id: str = "default",
                     dry_run: Optional[bool] = None, source: str = "workflow") -> GatewayResult:
        req = self.build_request(name, params)
        if req is None:
            return GatewayResult("failed", f"Workflow '{name}' not found.")
        missing = missing_params(req.args["steps"], params or {})
        if missing:
            return GatewayResult("failed", f"Workflow '{name}' needs values for: {', '.join(missing)}.")
        result = await gateway.execute(req, session_id, dry_run=dry_run, source=source)
        if result.status in ("executed", "failed"):
            workflow_recorder.record_run(name, result.status == "executed")
        return result


workflow_player = WorkflowPlayer()
