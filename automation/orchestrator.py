"""JARVIS Automation Orchestrator — Top-level entry point for the automation system.

Called by BrainManager when AUTOMATION_TASK intent is detected.
Routes: NL command → Task Plan → Execution → Result
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from automation.config import automation_config
from automation.automation_logger import automation_logger, log_action
from automation.action_executor import (
    trigger_emergency_stop,
    clear_emergency_stop,
    is_emergency_stopped,
)
from automation.task_manager import task_manager, AutomationTask, TaskStatus
from automation.workflow_recorder import workflow_recorder
from automation.workflow_player import workflow_player
from automation.screen_observer import screen_observer
from automation.safety_manager import safety_manager


@dataclass
class AutomationResult:
    """Result returned to BrainManager after automation execution."""
    success: bool
    summary: str
    task_id: str = ""
    elapsed_ms: float = 0.0
    steps_done: int = 0
    steps_failed: int = 0
    dry_run: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


# Natural language routing keywords
_RECORD_TRIGGERS = {"record", "start recording", "watch what i do", "learn this", "remember this"}
_STOP_RECORD_TRIGGERS = {"stop recording", "done recording", "finish recording", "end recording"}
_STOP_TRIGGERS = {"stop", "stop everything", "cancel", "abort", "emergency stop", "halt"}
_SCREENSHOT_TRIGGERS = {"screenshot", "take a screenshot", "capture screen", "what's on screen", "show screen"}
_STATUS_TRIGGERS = {"status", "what are you doing", "automation status"}
_DRY_RUN_TRIGGERS = {"dry run", "test run", "simulate", "don't actually do it"}
_REPLAY_TRIGGERS = {"replay", "run workflow", "play workflow", "repeat", "do again"}
_LIST_TRIGGERS = {"list workflows", "show workflows", "what workflows", "saved workflows"}


class AutomationOrchestrator:
    """
    Top-level automation controller.
    Interprets NL commands and routes to appropriate subsystems.
    """

    def __init__(self) -> None:
        self._dry_run_mode = False

    async def execute(self, command: str, session_id: str = "") -> AutomationResult:
        """
        Main entry point called by BrainManager.
        Parses intent → routes → returns AutomationResult.
        """
        cmd = command.lower().strip()
        start = time.perf_counter()

        automation_logger.info(f"COMMAND: '{command}'")

        # ── Emergency Stop ────────────────────────────────────────────
        if any(t in cmd for t in _STOP_TRIGGERS) and "recording" not in cmd:
            trigger_emergency_stop("user command")
            return AutomationResult(
                success=True,
                summary="[STOP] Emergency stop triggered. All automation halted.",
                dry_run=self._dry_run_mode,
            )

        # Clear emergency stop if user wants to resume
        if is_emergency_stopped() and "stop" not in cmd:
            clear_emergency_stop()

        # ── Dry-run toggle ────────────────────────────────────────────
        if any(t in cmd for t in _DRY_RUN_TRIGGERS):
            self._dry_run_mode = not self._dry_run_mode
            automation_config.DRY_RUN = self._dry_run_mode
            status = "enabled" if self._dry_run_mode else "disabled"
            return AutomationResult(
                success=True,
                summary=f"[TEST] Dry-run mode {status}. Actions will {'be simulated' if self._dry_run_mode else 'execute for real'}.",
                dry_run=self._dry_run_mode,
            )

        # ── Screenshot ────────────────────────────────────────────────
        if any(t in cmd for t in _SCREENSHOT_TRIGGERS):
            return await self._handle_screenshot(session_id)

        # ── Status ────────────────────────────────────────────────────
        if any(t in cmd for t in _STATUS_TRIGGERS):
            return self._handle_status()

        # ── List workflows ────────────────────────────────────────────
        if any(t in cmd for t in _LIST_TRIGGERS):
            workflows = workflow_recorder.list_workflows()
            names = ", ".join(workflows) if workflows else "none"
            return AutomationResult(
                success=True,
                summary=f"[LIST] Saved workflows: {names}",
                details={"workflows": workflows},
            )

        # ── Start recording ───────────────────────────────────────────
        if any(t in cmd for t in _RECORD_TRIGGERS):
            # Extract workflow name from command
            name = self._extract_workflow_name(command) or "unnamed_workflow"
            result = workflow_recorder.start_recording(name, description=command)
            return AutomationResult(
                success=result["success"],
                summary=f"[REC] Recording started: '{name}'. Perform actions — I'll watch and learn.",
                details=result,
            )

        # ── Stop recording ────────────────────────────────────────────
        if any(t in cmd for t in _STOP_RECORD_TRIGGERS):
            if workflow_recorder.is_recording:
                stop_result = workflow_recorder.stop_recording()
                save_result = workflow_recorder.save_workflow()
                steps = stop_result.get("steps", 0)
                name = save_result.get("name", "workflow")
                return AutomationResult(
                    success=True,
                    summary=f"[SAVED] Recording saved: '{name}' ({steps} steps). Say 'replay {name}' to run it.",
                    details=save_result,
                )
            return AutomationResult(success=False, summary="No active recording to stop.")

        # ── Replay workflow ────────────────────────────────────────────
        if any(t in cmd for t in _REPLAY_TRIGGERS):
            name = self._extract_workflow_name(command)
            if not name:
                workflows = workflow_recorder.list_workflows()
                if workflows:
                    name = workflows[-1]  # replay most recent
                else:
                    return AutomationResult(success=False, summary="No saved workflows found to replay.")

            replay_result = await workflow_player.replay(name)
            elapsed = (time.perf_counter() - start) * 1000
            success = replay_result.get("success", False)
            done = replay_result.get("steps_total", 0) - replay_result.get("steps_failed", 0)
            total = replay_result.get("steps_total", 0)
            status_tag = "[OK]" if success else "[WARN]"
            return AutomationResult(
                success=success,
                summary=(
                    f"{status_tag} Workflow '{name}' replayed: "
                    f"{done}/{total} steps succeeded."
                ),
                elapsed_ms=elapsed,
                details=replay_result,
            )

        # ── General automation task ────────────────────────────────────
        return await self._run_task(command, session_id, start)

    async def _run_task(
        self, command: str, session_id: str, start: float
    ) -> AutomationResult:
        """Plan and execute a general automation task from natural language."""
        try:
            # Step 1: Plan
            task = await task_manager.plan_from_nl(command, session_id)

            if not task.steps:
                return AutomationResult(
                    success=False,
                    summary="I couldn't create an automation plan for that command.",
                )

            # Step 2: Execute
            completed_task = await task_manager.execute_task(task)
            elapsed = (time.perf_counter() - start) * 1000

            done = len([s for s in completed_task.steps if s.status == "done"])
            failed = len([s for s in completed_task.steps if s.status == "failed"])
            success = completed_task.status == TaskStatus.COMPLETED

            return AutomationResult(
                success=success,
                summary=completed_task.result_summary,
                task_id=completed_task.id,
                elapsed_ms=round(elapsed, 1),
                steps_done=done,
                steps_failed=failed,
                dry_run=automation_config.DRY_RUN,
                details={
                    "goal": completed_task.goal,
                    "status": completed_task.status,
                    "steps": [
                        {
                            "index": s.index,
                            "action": s.action,
                            "target": s.target,
                            "status": s.status,
                            "error": s.error,
                        }
                        for s in completed_task.steps
                    ],
                },
            )

        except Exception as e:
            automation_logger.error(f"Task execution error: {e}")
            elapsed = (time.perf_counter() - start) * 1000
            return AutomationResult(
                success=False,
                summary=f"Automation failed: {str(e)}",
                elapsed_ms=round(elapsed, 1),
            )

    async def _handle_screenshot(self, session_id: str) -> AutomationResult:
        """Capture and describe the current screen."""
        try:
            state = await screen_observer.analyze(
                "Describe what is currently visible on the screen.",
                session_id=session_id or "automation",
            )
            desc = state.vision_description or state.ocr_text[:300] or "Screen captured."
            return AutomationResult(
                success=True,
                summary=f"[SCREEN] {desc}",
                details={
                    "active_app": state.active_app,
                    "window_title": state.window_title,
                    "ocr_preview": state.ocr_text[:200],
                },
            )
        except Exception as e:
            return AutomationResult(success=False, summary=f"Screenshot failed: {e}")

    def _handle_status(self) -> AutomationResult:
        """Return current automation status."""
        status = task_manager.get_status()
        workflows = workflow_recorder.list_workflows()
        is_recording = workflow_recorder.is_recording
        dry = automation_config.DRY_RUN

        lines = ["**JARVIS Automation Status:**"]
        lines.append(f"- Task running: {'Yes — ' + status.get('goal', '') if status.get('running') else 'No'}")
        lines.append(f"- Recording: {'🔴 YES' if is_recording else 'No'}")
        lines.append(f"- Dry-run mode: {'🧪 ON' if dry else 'Off'}")
        lines.append(f"- Saved workflows: {len(workflows)} ({', '.join(workflows[:3])}{'...' if len(workflows) > 3 else ''})")
        lines.append(f"- Emergency stop: {'🛑 ACTIVE' if is_emergency_stopped() else 'Clear'}")

        return AutomationResult(
            success=True,
            summary="\n".join(lines),
            details=status,
        )

    def _extract_workflow_name(self, command: str) -> Optional[str]:
        """Extract workflow name from commands like 'replay gmail login'."""
        import re
        # Remove common prefixes
        clean = re.sub(
            r"^(replay|run workflow|play workflow|record|start recording|learn|remember|watch me)\s+",
            "", command.lower().strip()
        )
        # Remove trailing articles
        clean = re.sub(r"\s+(workflow|recording)$", "", clean).strip()
        return clean if clean else None


# Global singleton
automation_orchestrator = AutomationOrchestrator()
