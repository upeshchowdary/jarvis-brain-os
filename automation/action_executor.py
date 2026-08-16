"""JARVIS Action Executor — Central VALIDATE → EXECUTE → OBSERVE → VERIFY loop.

Every automation action goes through this engine.
Provides: retry logic, emergency stop, dry-run, pre/post screenshot, and verification.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

from automation.config import automation_config
from automation.automation_logger import log_action, log_task_start, log_task_complete
from automation.safety_manager import safety_manager

# Emergency stop signal — shared across entire automation system
_emergency_stop_event = asyncio.Event()


def trigger_emergency_stop(reason: str = "user request") -> None:
    """Trigger immediate stop of all running automation."""
    from automation.automation_logger import log_emergency_stop
    log_emergency_stop(reason)
    _emergency_stop_event.set()


def clear_emergency_stop() -> None:
    """Reset emergency stop so new tasks can run."""
    _emergency_stop_event.clear()


def is_emergency_stopped() -> bool:
    return _emergency_stop_event.is_set()


@dataclass
class ActionResult:
    """Result of a single executed action."""
    action: str
    target: str
    success: bool
    result: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    retries: int = 0
    elapsed_ms: float = 0.0
    dry_run: bool = False
    blocked: bool = False
    confirmation_prompt: str = ""


class ActionExecutor:
    """
    Executes automation actions with:
    - Pre-action safety check
    - Emergency stop polling
    - Configurable retries with exponential backoff
    - Pre/post screenshot capture for verification
    - Structured ActionResult output
    """

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run or automation_config.DRY_RUN
        self._results: List[ActionResult] = []

    @property
    def results(self) -> List[ActionResult]:
        return self._results

    async def execute(
        self,
        action_name: str,
        coro_factory: Callable[[], Coroutine],
        target: str = "",
        max_retries: int | None = None,
        timeout: float | None = None,
        verify_fn: Optional[Callable[[], Coroutine]] = None,
        user_confirmed: bool = False,
        capture_screenshots: bool = False,
    ) -> ActionResult:
        """
        Execute a single action with full lifecycle management.

        Args:
            action_name:     Human-readable name of the action
            coro_factory:    Callable that returns an awaitable (the actual action)
            target:          Description of what is being acted on
            max_retries:     Override default retry count
            timeout:         Override default action timeout
            verify_fn:       Optional async verification coroutine
            user_confirmed:  Whether user has explicitly confirmed a risky action
            capture_screenshots: Whether to capture before/after screenshots
        """
        # ── Emergency stop check ────────────────────────────────────
        if is_emergency_stopped():
            return ActionResult(
                action=action_name, target=target,
                success=False, error="Emergency stop is active.",
            )

        # ── Safety check ─────────────────────────────────────────────
        allowed, reason = safety_manager.is_allowed(action_name, target, user_confirmed)
        if not allowed:
            prompt = safety_manager.format_confirmation_request(action_name, target)
            log_action("executor", action_name, target, result=f"BLOCKED: {reason}", level="WARNING")
            return ActionResult(
                action=action_name, target=target,
                success=False, blocked=True,
                error=reason, confirmation_prompt=prompt,
            )

        retries = max_retries if max_retries is not None else automation_config.MAX_RETRIES
        timeout_s = timeout or automation_config.ACTION_TIMEOUT
        pre_screenshot = None
        start = time.perf_counter()

        # ── Pre-action screenshot ────────────────────────────────────
        if capture_screenshots:
            try:
                from automation.screen_observer import screen_observer
                pre_screenshot = screen_observer.capture()
            except Exception:
                pass

        # ── Execute with retries ─────────────────────────────────────
        last_result = {}
        last_error = ""

        for attempt in range(retries + 1):
            if is_emergency_stopped():
                last_error = "Emergency stop triggered during retry."
                break

            try:
                log_action("executor", action_name, target,
                          result=f"attempt {attempt + 1}/{retries + 1}",
                          retry=attempt, dry_run=self.dry_run)

                result_data = await asyncio.wait_for(coro_factory(), timeout=timeout_s)
                last_result = result_data or {}

                if last_result.get("success", True):
                    break  # Success — stop retrying

                last_error = last_result.get("error", "Unknown error")
                log_action("executor", action_name, target,
                          result=f"FAILED attempt {attempt + 1}: {last_error}",
                          level="WARNING", retry=attempt)

            except asyncio.TimeoutError:
                last_error = f"Action timed out after {timeout_s}s"
                log_action("executor", action_name, target,
                          result=f"TIMEOUT attempt {attempt + 1}",
                          level="WARNING", retry=attempt)

            except Exception as e:
                last_error = str(e)
                log_action("executor", action_name, target,
                          result=f"EXCEPTION: {e}",
                          level="ERROR", retry=attempt)

            # Exponential backoff before retry
            if attempt < retries:
                wait = 0.5 * (2 ** attempt)
                await asyncio.sleep(wait)

        elapsed = (time.perf_counter() - start) * 1000
        success = last_result.get("success", False) and not last_error

        # ── Post-action verification ──────────────────────────────────
        if success and verify_fn:
            try:
                verified = await asyncio.wait_for(verify_fn(), timeout=automation_config.VERIFICATION_TIMEOUT)
                if not verified:
                    success = False
                    last_error = "Post-action verification failed."
                    log_action("executor", action_name, target, result="VERIFY FAILED", level="WARNING")
            except Exception as e:
                log_action("executor", action_name, target,
                          result=f"VERIFY ERROR: {e}", level="WARNING")

        # ── Emit final result ─────────────────────────────────────────
        ar = ActionResult(
            action=action_name,
            target=target,
            success=success,
            result=last_result,
            error=last_error,
            retries=attempt,
            elapsed_ms=round(elapsed, 1),
            dry_run=self.dry_run,
        )
        self._results.append(ar)

        level = "INFO" if success else "WARNING"
        log_action(
            "executor", action_name, target,
            result=f"{'OK' if success else 'FAILED'} ({elapsed:.0f}ms, retries={attempt})",
            level=level,
        )
        return ar

    def reset(self) -> None:
        """Clear result history."""
        self._results.clear()


# Global singleton
action_executor = ActionExecutor()
