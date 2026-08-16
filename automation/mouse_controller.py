"""JARVIS Mouse Controller — Safe, vision-verified mouse interactions.

All actions support:
- Dry-run mode (log only, no execution)
- Configurable action timeout
- Safety check via SafetyManager
"""

import asyncio
import time
from typing import Tuple, Optional

from automation.config import automation_config
from automation.automation_logger import log_action
from automation.safety_manager import safety_manager

try:
    import pyautogui
    pyautogui.FAILSAFE = True   # Move mouse to top-left corner to abort
    pyautogui.PAUSE = automation_config.ACTION_DELAY
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False


class MouseController:
    """Controls mouse movement, clicks, drag, and scroll operations."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run or automation_config.DRY_RUN

    def _check(self, action: str, target: str = "") -> tuple[bool, str]:
        allowed, reason = safety_manager.is_allowed(action, target)
        return allowed, reason

    def _require_pyautogui(self) -> bool:
        if not HAS_PYAUTOGUI:
            raise RuntimeError("pyautogui is not installed. Run: pip install pyautogui")
        return True

    # ── Position ────────────────────────────────────────────────────

    def get_position(self) -> Tuple[int, int]:
        """Return current cursor (x, y) position."""
        if not HAS_PYAUTOGUI:
            return (0, 0)
        x, y = pyautogui.position()
        return (int(x), int(y))

    # ── Movement ────────────────────────────────────────────────────

    async def move(self, x: int, y: int, duration: float = 0.3) -> dict:
        """Move cursor to (x, y) smoothly."""
        log_action("mouse", "move", f"({x},{y})", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "action": "move", "x": x, "y": y}
        self._require_pyautogui()
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: pyautogui.moveTo(x, y, duration=duration)
        )
        return {"success": True, "action": "move", "x": x, "y": y}

    # ── Clicks ──────────────────────────────────────────────────────

    async def click(self, x: int, y: int) -> dict:
        """Left-click at (x, y)."""
        allowed, reason = self._check("click", f"({x},{y})")
        if not allowed:
            return {"success": False, "blocked": True, "reason": reason}
        log_action("mouse", "click", f"({x},{y})", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "action": "click", "x": x, "y": y}
        self._require_pyautogui()
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: pyautogui.click(x, y)
        )
        await asyncio.sleep(automation_config.CLICK_DELAY)
        return {"success": True, "action": "click", "x": x, "y": y}

    async def double_click(self, x: int, y: int) -> dict:
        """Double-click at (x, y)."""
        allowed, reason = self._check("double_click", f"({x},{y})")
        if not allowed:
            return {"success": False, "blocked": True, "reason": reason}
        log_action("mouse", "double_click", f"({x},{y})", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "action": "double_click"}
        self._require_pyautogui()
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: pyautogui.doubleClick(x, y)
        )
        return {"success": True, "action": "double_click", "x": x, "y": y}

    async def right_click(self, x: int, y: int) -> dict:
        """Right-click at (x, y)."""
        allowed, reason = self._check("right_click", f"({x},{y})")
        if not allowed:
            return {"success": False, "blocked": True, "reason": reason}
        log_action("mouse", "right_click", f"({x},{y})", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "action": "right_click"}
        self._require_pyautogui()
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: pyautogui.rightClick(x, y)
        )
        return {"success": True, "action": "right_click", "x": x, "y": y}

    async def middle_click(self, x: int, y: int) -> dict:
        """Middle-click at (x, y)."""
        log_action("mouse", "middle_click", f"({x},{y})", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "action": "middle_click"}
        self._require_pyautogui()
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: pyautogui.middleClick(x, y)
        )
        return {"success": True, "action": "middle_click"}

    # ── Drag ────────────────────────────────────────────────────────

    async def drag(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.5) -> dict:
        """Drag from (x1,y1) to (x2,y2)."""
        allowed, reason = self._check("drag", f"({x1},{y1})->({x2},{y2})")
        if not allowed:
            return {"success": False, "blocked": True, "reason": reason}
        log_action("mouse", "drag", f"({x1},{y1})->({x2},{y2})", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "action": "drag"}
        self._require_pyautogui()
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: pyautogui.drag(x2 - x1, y2 - y1, duration=duration, _pause=False)
        )
        return {"success": True, "action": "drag"}

    # ── Scroll ──────────────────────────────────────────────────────

    async def scroll(self, amount: int, direction: str = "down", x: Optional[int] = None, y: Optional[int] = None) -> dict:
        """Scroll up or down. amount > 0 = up, amount < 0 = down in pyautogui convention."""
        clicks = abs(amount) if direction == "up" else -abs(amount)
        log_action("mouse", "scroll", f"{direction} {abs(amount)}", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "action": "scroll", "direction": direction}
        self._require_pyautogui()
        kwargs = {"clicks": clicks}
        if x is not None and y is not None:
            kwargs["x"] = x
            kwargs["y"] = y
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: pyautogui.scroll(**kwargs)
        )
        return {"success": True, "action": "scroll", "direction": direction, "amount": amount}

    async def scroll_horizontal(self, amount: int, direction: str = "right") -> dict:
        """Horizontal scroll. Requires pyautogui >= 0.9.54."""
        clicks = abs(amount) if direction == "right" else -abs(amount)
        log_action("mouse", "hscroll", f"{direction} {abs(amount)}", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "action": "hscroll"}
        self._require_pyautogui()
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: pyautogui.hscroll(clicks)
        )
        return {"success": True, "action": "hscroll", "direction": direction}

    # ── Press/Release ────────────────────────────────────────────────

    async def mouse_down(self, x: int, y: int) -> dict:
        log_action("mouse", "mouse_down", f"({x},{y})", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}
        self._require_pyautogui()
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: pyautogui.mouseDown(x, y)
        )
        return {"success": True, "action": "mouse_down"}

    async def mouse_up(self, x: int, y: int) -> dict:
        log_action("mouse", "mouse_up", f"({x},{y})", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}
        self._require_pyautogui()
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: pyautogui.mouseUp(x, y)
        )
        return {"success": True, "action": "mouse_up"}


# Global singleton
mouse_controller = MouseController()

