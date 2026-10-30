"""Computer-use orchestrator: plan → (preview + confirm via gateway) → act/verify/recover.

The orchestrator never executes anything itself: it turns a goal into a
``computer.run_plan`` request and hands it to the ActionGateway, which applies
the risk policy, dry-run, confirmation and audit logging.
"""

from __future__ import annotations

from typing import Optional

from automation.automation_logger import automation_logger
from automation.engine.planner import task_planner
from automation.gateway import ActionRequest, GatewayResult, gateway


class AdvancedOrchestrator:
    async def plan_request(self, goal: str) -> Optional[ActionRequest]:
        steps = await task_planner.plan(goal)
        if not steps:
            return None
        return ActionRequest("computer.run_plan", {"goal": goal, "steps": steps}, description=goal)

    async def execute(self, goal: str, session_id: str = "default", *, dry_run: Optional[bool] = None,
                      source: str = "chat") -> GatewayResult:
        automation_logger.info(f"[Orchestrator] Planning: '{goal}'")
        request = await self.plan_request(goal)
        if request is None:
            return GatewayResult("failed", "I couldn't work out the steps for that. Could you describe it differently?")
        return await gateway.execute(request, session_id, dry_run=dry_run, source=source)


advanced_orchestrator = AdvancedOrchestrator()
