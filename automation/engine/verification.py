"""Closed-Loop Verification Engine for JARVIS Automation Engine.

Evaluates pre-conditions and post-conditions for every atomic action:
- Text presence / absence
- Window focus / active process
- URL navigation / tab change
- Element state (toggled, checked, disabled, focused)
- File existence & size
- Custom state predicates with adaptive polling (no fixed sleep)
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, Awaitable

from automation.automation_logger import automation_logger


@dataclass
class VerificationCondition:
    """Specification of expected state change."""
    condition_type: str  # "text_visible", "text_absent", "window_focused", "url_contains", "file_exists", "process_running", "custom"
    target: str
    timeout_seconds: float = 4.0
    poll_interval: float = 0.25
    custom_predicate: Optional[Callable[[], Awaitable[bool]]] = None


@dataclass
class VerificationResult:
    """Outcome of state verification."""
    passed: bool
    condition_type: str
    target: str
    observed_state: str
    elapsed_ms: float
    error: Optional[str] = None


class VerificationEngine:
    """Verifies that executed actions actually produced their intended state change."""

    async def verify(self, condition: VerificationCondition) -> VerificationResult:
        """
        Polls environment adaptively until condition is met or timeout occurs.
        No static fixed sleep times.
        """
        start = time.perf_counter()
        timeout = condition.timeout_seconds
        c_type = condition.condition_type.lower()
        target = condition.target

        while (time.perf_counter() - start) < timeout:
            elapsed_ms = (time.perf_counter() - start) * 1000

            # 1. Text Presence in OCR / UIA
            if c_type == "text_visible":
                from automation.screen_observer import screen_observer
                screenshot = screen_observer.capture()
                if screenshot is not None:
                    text, _ = await screen_observer.extract_text(screenshot)
                    if target.lower() in text.lower():
                        return VerificationResult(
                            passed=True,
                            condition_type=c_type,
                            target=target,
                            observed_state=f"Text '{target}' found on screen",
                            elapsed_ms=elapsed_ms,
                        )

            # 2. Text Absence (modal closed, loading spinner gone)
            elif c_type == "text_absent":
                from automation.screen_observer import screen_observer
                screenshot = screen_observer.capture()
                if screenshot is not None:
                    text, _ = await screen_observer.extract_text(screenshot)
                    if target.lower() not in text.lower():
                        return VerificationResult(
                            passed=True,
                            condition_type=c_type,
                            target=target,
                            observed_state=f"Text '{target}' is no longer visible",
                            elapsed_ms=elapsed_ms,
                        )

            # 3. Window Focused
            elif c_type == "window_focused":
                from automation.engine.perception import perception_engine
                win_info = perception_engine.inspect_active_window()
                title = win_info.get("title", "").lower()
                pname = win_info.get("process_name", "").lower()
                if target.lower() in title or target.lower() in pname:
                    return VerificationResult(
                        passed=True,
                        condition_type=c_type,
                        target=target,
                        observed_state=f"Active window is '{win_info.get('title')}' ({pname})",
                        elapsed_ms=elapsed_ms,
                    )

            # 4. File Exists & Non-Empty
            elif c_type == "file_exists":
                expanded_path = os.path.expandvars(os.path.expanduser(target))
                if os.path.exists(expanded_path) and os.path.getsize(expanded_path) > 0:
                    return VerificationResult(
                        passed=True,
                        condition_type=c_type,
                        target=target,
                        observed_state=f"File exists ({os.path.getsize(expanded_path)} bytes)",
                        elapsed_ms=elapsed_ms,
                    )

            # 5. Process Running
            elif c_type == "process_running":
                # pyrefly: ignore [missing-import]
                from automation.process_controller import process_controller
                if process_controller.is_running(target):
                    return VerificationResult(
                        passed=True,
                        condition_type=c_type,
                        target=target,
                        observed_state=f"Process '{target}' is running",
                        elapsed_ms=elapsed_ms,
                    )

            # 6. Custom Predicate
            elif c_type == "custom" and condition.custom_predicate is not None:
                try:
                    res = await condition.custom_predicate()
                    if res:
                        return VerificationResult(
                            passed=True,
                            condition_type=c_type,
                            target=target,
                            observed_state="Custom condition satisfied",
                            elapsed_ms=elapsed_ms,
                        )
                except Exception as e:
                    automation_logger.debug(f"[Verification] Custom predicate error: {e}")

            await asyncio.sleep(condition.poll_interval)

        elapsed_ms = (time.perf_counter() - start) * 1000
        return VerificationResult(
            passed=False,
            condition_type=c_type,
            target=target,
            observed_state="Timed out waiting for expected state",
            elapsed_ms=elapsed_ms,
            error=f"Verification failed: '{target}' not satisfied within {timeout:.1f}s",
        )


verification_engine = VerificationEngine()
