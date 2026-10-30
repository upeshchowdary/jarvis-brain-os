"""Advanced Execution Orchestrator for JARVIS Automation Engine.

Implements the central closed-loop execution loop:
OBSERVE -> PLAN -> ACT -> VERIFY -> (RECOVER) -> ADAPT -> COMPLETE
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Any

from automation.automation_logger import automation_logger
from automation.engine.action_router import action_router
from automation.engine.context_memory import execution_memory
from automation.engine.perception import perception_engine
from automation.engine.planner import task_planner
from automation.engine.recovery import recovery_engine
from automation.engine.verification import verification_engine


@dataclass
class EngineExecutionResult:
    """Final outcome of an automation workflow."""
    success: bool
    summary: str
    goal: str
    steps_total: int
    steps_completed: int
    elapsed_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


class AdvancedOrchestrator:
    """Central closed-loop automation orchestrator."""

    async def execute(self, user_instruction: str, session_id: str = "default") -> EngineExecutionResult:
        """
        Executes a natural language instruction through the full 7-subsystem pipeline.
        """
        start_time = time.perf_counter()
        automation_logger.info(f"[Orchestrator] Starting task: '{user_instruction}'")

        # ── 1. Fast Check for Specialized Communication (Email / WhatsApp / Telegram / Discord / Slack) ──
        import re
        q_low = user_instruction.lower()
        has_email_addr = bool(re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", user_instruction))
        has_comm_app = bool(re.search(r"\b(?:whatsapp|wa|wapp|whats\s*app|telegram|tg|tele|discord|slack|gmail|email|e-mail|mail)\b", q_low))
        has_comm_verb = bool(re.search(r"\b(?:message|msg|text|ask|tell|say|send|invite|dm|write|compose|mail|email)\b", q_low))
        is_comm_intent = (has_comm_app or has_email_addr) and has_comm_verb

        if is_comm_intent:
            from automation.engine.messaging_handler import extract_messaging_intent, dispatch_message
            contact, msg_text, app, subject = await extract_messaging_intent(user_instruction)
            if contact and msg_text:
                msg_res = await dispatch_message(app, contact, msg_text, subject=subject)
                elapsed = (time.perf_counter() - start_time) * 1000
                summary = msg_res.get("summary", f"{app.title()} communication sent to {contact}")
                execution_memory.record_action(f"{app}_communication", user_instruction, summary, msg_res.get("success", False))
                return EngineExecutionResult(
                    success=msg_res.get("success", False),
                    summary=summary,
                    goal=user_instruction,
                    steps_total=1,
                    steps_completed=1 if msg_res.get("success") else 0,
                    elapsed_ms=elapsed,
                    details=msg_res,
                )

        # ── 2. Environment Perception ─────────────────────────────────────────
        current_state = await perception_engine.capture_state(include_visual=False)
        execution_memory.update_app_state(
            active_app=current_state.active_process_name,
            window_title=current_state.active_window_title,
        )

        # ── 3. Plan Decomposition ─────────────────────────────────────────────
        task_graph = await task_planner.plan(user_instruction)
        automation_logger.info(f"[Orchestrator] Generated plan with {len(task_graph.nodes)} steps for '{user_instruction}'")

        steps_done = 0
        total_steps = len(task_graph.nodes)

        # ── 4. Closed-Loop Step Execution Loop ────────────────────────────────
        for idx, node in enumerate(task_graph.nodes):
            automation_logger.info(f"[Orchestrator] Step {idx+1}/{total_steps}: {node.description}")

            # ACT: Dispatch to ActionRouter
            action_res = await action_router.dispatch(node.action_spec)

            # VERIFY: If node has verification condition, evaluate it
            verified = action_res.success
            if action_res.success and node.verification is not None:
                v_res = await verification_engine.verify(node.verification)
                verified = v_res.passed
                if not verified:
                    automation_logger.warning(f"[Orchestrator] Step verification failed: {v_res.error}")

            # RECOVER: If failed or unverified, engage RecoveryEngine
            if not verified:
                automation_logger.info(f"[Orchestrator] Engaging Recovery Engine for step '{node.description}'")
                recovered_res = await recovery_engine.escalate_and_retry(
                    action=node.action_spec,
                    expected_condition=node.verification,
                )
                if recovered_res.success:
                    verified = True
                    action_res = recovered_res

            if verified:
                node.is_completed = True
                steps_done += 1
                execution_memory.record_action(node.action_spec.action_type, node.action_spec.target, action_res.output_data, True)
            else:
                node.is_failed = True
                execution_memory.record_action(node.action_spec.action_type, node.action_spec.target, action_res.error, False)
                automation_logger.error(f"[Orchestrator] Step failed after recovery: {action_res.error}")
                # Abort or adapt
                break

        elapsed_total = (time.perf_counter() - start_time) * 1000
        success = (steps_done == total_steps and total_steps > 0)

        summary = (
            f"Completed '{user_instruction}' ({steps_done}/{total_steps} steps succeeded in {elapsed_total:.0f}ms)"
            if success else
            f"Partially completed '{user_instruction}' ({steps_done}/{total_steps} steps executed in {elapsed_total:.0f}ms)"
        )

        return EngineExecutionResult(
            success=success,
            summary=summary,
            goal=user_instruction,
            steps_total=total_steps,
            steps_completed=steps_done,
            elapsed_ms=elapsed_total,
            details={"plan_nodes": [n.description for n in task_graph.nodes]},
        )


advanced_orchestrator = AdvancedOrchestrator()
