"""JARVIS Workflow Player — Adaptive Vision-guided workflow replay.

Does NOT blindly replay coordinates.
For each step: observe screen -> find equivalent target -> execute -> verify.
"""

import asyncio
from typing import Optional, Dict, Any, List

from automation.config import automation_config
from automation.automation_logger import log_action, automation_logger
from automation.workflow_recorder import Workflow, WorkflowStep, workflow_recorder
from automation.action_executor import action_executor, is_emergency_stopped
from automation.screen_observer import screen_observer
from automation.mouse_controller import mouse_controller
from automation.keyboard_controller import keyboard_controller
from automation.window_controller import window_controller
from automation.application_controller import application_controller
from automation.browser_controller import browser_controller


class WorkflowPlayer:
    """Adaptively replays recorded workflows using Vision-guided targeting."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run or automation_config.DRY_RUN

    async def replay(
        self,
        workflow_name: str,
        adaptive: bool = True,
    ) -> Dict[str, Any]:
        """
        Replay a named workflow.

        Args:
            workflow_name: Name of saved workflow to replay
            adaptive:      If True, use Vision to find targets (ignores saved coordinates)
                          If False, use saved coordinates directly (faster but fragile)
        """
        wf = workflow_recorder.load_workflow(workflow_name)
        if not wf:
            return {"success": False, "error": f"Workflow '{workflow_name}' not found."}

        automation_logger.info(f"REPLAY START: '{wf.name}' ({len(wf.steps)} steps, adaptive={adaptive})")
        results = []
        failed_steps = []

        for step in wf.steps:
            if is_emergency_stopped():
                automation_logger.warning("REPLAY ABORTED: Emergency stop triggered.")
                break

            log_action("player", f"step[{step.step_index}]",
                      f"{step.action} -> {step.target or step.value}")

            result = await self._execute_step(step, adaptive=adaptive)
            results.append({
                "step": step.step_index,
                "action": step.action,
                "target": step.target or step.value,
                "success": result.get("success", False),
                "error": result.get("error", ""),
            })

            if not result.get("success"):
                failed_steps.append(step.step_index)
                # Try recovery: re-observe and retry once
                if adaptive:
                    automation_logger.warning(
                        f"Step {step.step_index} failed — attempting adaptive recovery..."
                    )
                    await asyncio.sleep(1.0)
                    recovery = await self._execute_step(step, adaptive=True)
                    if recovery.get("success"):
                        results[-1]["success"] = True
                        results[-1]["recovered"] = True
                        failed_steps.pop()

            # Brief pause between steps
            await asyncio.sleep(automation_config.ACTION_DELAY)

        # Update run statistics
        wf.run_count += 1
        import time
        wf.last_run = time.time()
        wf.success_rate = 1.0 - (len(failed_steps) / max(len(wf.steps), 1))
        try:
            workflow_recorder.save_workflow()
        except Exception:
            pass

        success = len(failed_steps) == 0
        automation_logger.info(
            f"REPLAY DONE: '{wf.name}' | success={success} | "
            f"failed_steps={failed_steps}"
        )
        return {
            "success": success,
            "workflow": wf.name,
            "steps_total": len(wf.steps),
            "steps_failed": len(failed_steps),
            "failed_at": failed_steps,
            "results": results,
        }

    async def _execute_step(
        self, step: WorkflowStep, adaptive: bool = True
    ) -> Dict[str, Any]:
        """Execute a single workflow step, optionally using Vision to find targets."""
        action = step.action.lower()

        # ── Application open ─────────────────────────────────────────
        if action == "open_app":
            return await application_controller.open_app(step.target)

        # ── Navigation ───────────────────────────────────────────────
        if action == "navigate":
            return await browser_controller.navigate(step.target or step.value)

        # ── Click ────────────────────────────────────────────────────
        if action in ("click", "left_click"):
            if adaptive and step.target:
                # Find target by text using OCR
                loc = await screen_observer.find_text_location(step.target)
                if loc:
                    return await mouse_controller.click(loc[0], loc[1])
                # Fall back to Vision
                loc = await screen_observer.find_element_by_description(step.target)
                if loc:
                    return await mouse_controller.click(loc[0], loc[1])
                return {"success": False, "error": f"Could not find target: '{step.target}'"}
            elif step.x is not None and step.y is not None:
                return await mouse_controller.click(step.x, step.y)
            return {"success": False, "error": "No target or coordinates for click."}

        # ── Double click ─────────────────────────────────────────────
        if action == "double_click":
            if adaptive and step.target:
                loc = await screen_observer.find_text_location(step.target)
                if loc:
                    return await mouse_controller.double_click(loc[0], loc[1])
            elif step.x and step.y:
                return await mouse_controller.double_click(step.x, step.y)
            return {"success": False, "error": "No target for double_click."}

        # ── Typing ───────────────────────────────────────────────────
        if action in ("type", "type_text"):
            return await keyboard_controller.type_text(step.value)

        # ── Key press ────────────────────────────────────────────────
        if action in ("press", "press_key"):
            return await keyboard_controller.press(step.value)

        # ── Hotkey ───────────────────────────────────────────────────
        if action == "hotkey":
            keys = step.value.split("+") if "+" in step.value else [step.value]
            return await keyboard_controller.hotkey(*keys)

        # ── Scroll ───────────────────────────────────────────────────
        if action == "scroll":
            direction = step.value if step.value in ("up", "down") else "down"
            amount = int(step.target) if step.target.isdigit() else 3
            return await mouse_controller.scroll(amount, direction=direction)

        # ── Focus window ─────────────────────────────────────────────
        if action in ("focus_window", "switch_window"):
            return await window_controller.focus_window(step.target)

        # ── Fill field ───────────────────────────────────────────────
        if action == "fill_field":
            return await browser_controller.fill_field(step.target, step.value)

        # ── Wait ─────────────────────────────────────────────────────
        if action == "wait":
            try:
                secs = float(step.value or "1.0")
                await asyncio.sleep(secs)
                return {"success": True}
            except ValueError:
                return {"success": True}

        automation_logger.warning(f"Unknown step action: '{action}' — skipping.")
        return {"success": True, "skipped": True, "reason": f"Unknown action: {action}"}


# Global singleton
workflow_player = WorkflowPlayer()

