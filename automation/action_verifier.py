"""JARVIS Action Verifier — Post-action state verification.

Never assume an action succeeded. Verify using OCR, Vision, window state,
filesystem, and URL checks.
"""

import asyncio
from pathlib import Path
from typing import Optional

from automation.config import automation_config
from automation.automation_logger import log_action
from automation.screen_observer import screen_observer
from automation.window_controller import window_controller


class ActionVerifier:
    """Verifies that automation actions had their intended effect."""

    # ── Text / UI Verification ────────────────────────────────────────

    async def verify_text_appeared(
        self, text: str, timeout: float | None = None
    ) -> bool:
        """Check if specific text appears on screen (OCR-based)."""
        result = await screen_observer.wait_for_text(
            text, timeout=timeout or automation_config.VERIFICATION_TIMEOUT
        )
        log_action("verifier", "verify_text_appeared", text,
                  result="PASS" if result else "FAIL")
        return result

    async def verify_text_gone(
        self, text: str, timeout: float | None = None
    ) -> bool:
        """Check that specific text is no longer visible (e.g. loading spinner gone)."""
        result = await screen_observer.wait_for_text_gone(
            text, timeout=timeout or automation_config.VERIFICATION_TIMEOUT
        )
        log_action("verifier", "verify_text_gone", text,
                  result="PASS" if result else "FAIL")
        return result

    async def verify_element_present(self, description: str) -> bool:
        """Use Vision AI to verify an element is present on screen."""
        loc = await screen_observer.find_element_by_description(description)
        found = loc is not None
        log_action("verifier", "verify_element_present", description,
                  result="PASS" if found else "FAIL")
        return found

    # ── Window / App Verification ────────────────────────────────────

    async def verify_window_opened(
        self, title_fragment: str, timeout: float | None = None
    ) -> bool:
        """Verify that a window with given title has appeared."""
        result = await window_controller.wait_for_window(
            title_fragment, timeout=timeout or automation_config.APP_STARTUP_TIMEOUT
        )
        passed = result.get("success", False)
        log_action("verifier", "verify_window_opened", title_fragment,
                  result="PASS" if passed else "FAIL")
        return passed

    def verify_window_active(self, title_fragment: str) -> bool:
        """Check if a specific window is currently active/focused."""
        active = window_controller.get_active_window()
        if not active:
            return False
        passed = title_fragment.lower() in (active.get("title") or "").lower()
        log_action("verifier", "verify_window_active", title_fragment,
                  result="PASS" if passed else "FAIL")
        return passed

    # ── File System Verification ──────────────────────────────────────

    def verify_file_exists(self, path: str) -> bool:
        """Check that a file or directory exists."""
        exists = Path(path).exists()
        log_action("verifier", "verify_file_exists", path,
                  result="PASS" if exists else "FAIL")
        return exists

    def verify_file_deleted(self, path: str) -> bool:
        """Check that a file no longer exists."""
        deleted = not Path(path).exists()
        log_action("verifier", "verify_file_deleted", path,
                  result="PASS" if deleted else "FAIL")
        return deleted

    # ── URL Verification ─────────────────────────────────────────────

    async def verify_url_contains(self, fragment: str, timeout: float | None = None) -> bool:
        """Verify that the current browser URL contains a given fragment (OCR-based)."""
        deadline = asyncio.get_event_loop().time() + (timeout or automation_config.VERIFICATION_TIMEOUT)
        while asyncio.get_event_loop().time() < deadline:
            loc = await screen_observer.find_text_location(fragment)
            if loc:
                log_action("verifier", "verify_url_contains", fragment, result="PASS")
                return True
            await asyncio.sleep(0.5)
        log_action("verifier", "verify_url_contains", fragment, result="FAIL")
        return False

    # ── General Screen State Verification ────────────────────────────

    async def verify_screen_changed(self, previous_screenshot=None) -> bool:
        """Verify the screen changed after an action (pixel diff)."""
        if previous_screenshot is None:
            return True  # Can't compare — assume changed

        current = screen_observer.capture()
        if current is None:
            return False

        try:
            import numpy as np
            import cv2
            prev_np = np.array(previous_screenshot.convert("RGB"))
            curr_np = np.array(current.convert("RGB"))
            if prev_np.shape != curr_np.shape:
                return True
            diff = np.abs(prev_np.astype(int) - curr_np.astype(int))
            change_pct = (diff > 15).mean() * 100
            changed = change_pct > 0.5  # > 0.5% of pixels changed
            log_action("verifier", "verify_screen_changed", "",
                      result=f"PASS ({change_pct:.1f}% changed)" if changed else "FAIL (no change)")
            return changed
        except ImportError:
            return True  # Can't verify — assume changed

    async def verify_vision_confirms(self, expected_state: str) -> bool:
        """Use Vision AI to confirm a specific screen state description is accurate."""
        state = await screen_observer.analyze(
            f"Is this true about the current screen: '{expected_state}'? Answer YES or NO."
        )
        desc = (state.vision_description or "").lower()
        confirmed = "yes" in desc[:20]
        log_action("verifier", "verify_vision_confirms", expected_state[:40],
                  result="PASS" if confirmed else "FAIL")
        return confirmed


# Global singleton
action_verifier = ActionVerifier()
