"""Recovery Engine for JARVIS Automation Engine.

Implements self-correction and multi-tier recovery strategies:
1. Modal / Overlay / Cookie banner auto-dismissal
2. Hierarchy Escalation (L2 DOM -> L3 UIA -> L4 Keyboard -> L5 Mouse -> L6 Vision)
3. Window focus re-acquisition
4. Dynamic Re-planning when environment state diverges
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Any

from automation.automation_logger import automation_logger
from automation.engine.action_router import action_router, ActionSpec, ActionLevel, ActionExecutionResult
from automation.engine.perception import perception_engine
from automation.engine.verification import VerificationCondition, verification_engine


class RecoveryEngine:
    """Detects and resolves obstacles, modals, focus drops, and execution failures."""

    # Common dismissal targets (cookie dialogs, promo modals, update notices)
    DISMISS_TARGETS = [
        "Accept all", "Accept cookies", "I agree", "Got it", "Dismiss",
        "Close", "Skip", "No thanks", "Maybe later", "Continue", "Not now",
    ]

    async def try_dismiss_overlays(self) -> bool:
        """Attempts to detect and dismiss common blocking modals/banners."""
        screen_state = await perception_engine.capture_state(include_visual=True)
        ocr_text = screen_state.full_ocr_text.lower()

        for target in self.DISMISS_TARGETS:
            if target.lower() in ocr_text:
                automation_logger.info(f"[Recovery] Detected overlay with text '{target}', attempting dismiss")
                # Try clicking via mouse locator
                action = ActionSpec(
                    level=ActionLevel.LEVEL_5_MOUSE,
                    action_type="click",
                    target=target,
                )
                res = await action_router.dispatch(action)
                if res.success:
                    await asyncio.sleep(0.5)
                    return True

        return False

    async def refocus_window(self, app_or_title: str) -> bool:
        """Restores focus to the intended application window."""
        from automation.window_controller import window_controller
        return await window_controller.focus_window(app_or_title)

    async def escalate_and_retry(
        self,
        action: ActionSpec,
        expected_condition: Optional[VerificationCondition] = None,
    ) -> ActionExecutionResult:
        """
        Escalates through the 6-level hierarchy if the initial level fails:
        L2 (DOM) -> L3 (UIA) -> L4 (Keyboard) -> L5 (Mouse/OCR) -> L6 (Vision)
        """
        levels_order = [
            ActionLevel.LEVEL_2_DOM_WEB,
            ActionLevel.LEVEL_3_WINDOWS_UIA,
            ActionLevel.LEVEL_4_KEYBOARD,
            ActionLevel.LEVEL_5_MOUSE,
            ActionLevel.LEVEL_6_VISION,
        ]

        # Start from current action level
        start_idx = 0
        for i, lvl in enumerate(levels_order):
            if lvl == action.level:
                start_idx = i
                break

        last_result = None
        for lvl in levels_order[start_idx:]:
            automation_logger.info(f"[Recovery] Escalating action '{action.action_type}:{action.target}' to {lvl.value}")
            escalated_spec = ActionSpec(
                level=lvl,
                action_type=action.action_type,
                target=action.target,
                value=action.value,
                parameters=action.parameters,
            )
            result = await action_router.dispatch(escalated_spec)
            last_result = result

            if result.success:
                # If verification condition provided, verify the state
                if expected_condition:
                    v_res = await verification_engine.verify(expected_condition)
                    if v_res.passed:
                        return result
                    else:
                        automation_logger.warning(f"[Recovery] Action succeeded on {lvl.value} but verification failed")
                else:
                    return result

            # If failed, try clearing any blocking modals before next escalation
            await self.try_dismiss_overlays()

        return last_result or ActionExecutionResult(
            success=False,
            level_used=action.level,
            action_type=action.action_type,
            target=action.target,
            elapsed_ms=0,
            error="All escalation levels exhausted",
        )

    async def dynamic_replan(self, goal: str, failed_step: str, current_state_desc: str) -> List[Dict[str, Any]]:
        """
        Queries LLM to generate recovery sub-steps when state diverges from plan.
        """
        from brain.model_manager import model_manager
        prompt = (
            f"You are the JARVIS Recovery Engine.\n"
            f"Goal: \"{goal}\"\n"
            f"Failed Action: \"{failed_step}\"\n"
            f"Current Computer State: \"{current_state_desc}\"\n\n"
            f"Provide a 1 to 3 step JSON recovery plan to get back on track.\n"
            f"JSON format:\n"
            f"[\n"
            f'  {{"action": "click|type|navigate|open_app|press", "target": "...", "value": "..."}}\n'
            f"]\n"
            f"Output ONLY valid JSON array."
        )

        try:
            res = await model_manager.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
            )
            raw = (res.get("content", "") if isinstance(res, dict) else "").strip()
            raw = raw.strip("```json").strip("```").strip()
            import json
            return json.loads(raw)
        except Exception as e:
            automation_logger.warning(f"[Recovery] Dynamic replan LLM error: {e}")
            return []


recovery_engine = RecoveryEngine()
