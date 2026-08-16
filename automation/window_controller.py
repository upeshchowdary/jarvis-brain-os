"""JARVIS Window Controller — Application window management via Windows API.

Uses pygetwindow + win32api for window listing, focus, resize, and state control.
Falls back to graceful errors when windows are not found.
"""

import asyncio
import sys
from typing import List, Optional

from automation.config import automation_config
from automation.automation_logger import log_action

try:
    import pygetwindow as gw
    HAS_GW = True
except ImportError:
    HAS_GW = False

# Reuse existing WindowInfo model from vision layer
from vision.environment import WindowInfo, BoundingBox


class WindowController:
    """Manages application windows: find, focus, minimize, maximize, close, list."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run or automation_config.DRY_RUN

    def _require_gw(self) -> None:
        if not HAS_GW:
            raise RuntimeError("pygetwindow is not installed. Run: pip install pygetwindow")

    # ── Discovery ────────────────────────────────────────────────────

    def list_windows(self) -> List[dict]:
        """Return all open windows as list of dicts."""
        if not HAS_GW:
            return []
        try:
            windows = gw.getAllWindows()
            return [
                {
                    "title": w.title,
                    "left": w.left,
                    "top": w.top,
                    "width": w.width,
                    "height": w.height,
                    "visible": w.visible,
                    "active": w.isActive,
                }
                for w in windows
                if w.title.strip()
            ]
        except Exception:
            return []

    def get_active_window(self) -> Optional[dict]:
        """Return info about the currently active/focused window."""
        if not HAS_GW:
            return None
        try:
            w = gw.getActiveWindow()
            if w:
                return {
                    "title": w.title,
                    "left": w.left,
                    "top": w.top,
                    "width": w.width,
                    "height": w.height,
                }
        except Exception:
            pass
        return None

    def find_window(self, title_fragment: str) -> Optional[object]:
        """Find a window by partial title match."""
        if not HAS_GW:
            return None
        try:
            matches = gw.getWindowsWithTitle(title_fragment)
            return matches[0] if matches else None
        except Exception:
            return None

    def is_window_open(self, title_fragment: str) -> bool:
        """Check if a window containing title_fragment is currently open."""
        return self.find_window(title_fragment) is not None

    # ── State Control ────────────────────────────────────────────────

    async def focus_window(self, title_fragment: str) -> dict:
        """Bring a window to the foreground and focus it."""
        log_action("window", "focus", title_fragment, dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "action": "focus_window"}
        self._require_gw()
        w = self.find_window(title_fragment)
        if not w:
            return {"success": False, "error": f"Window '{title_fragment}' not found."}
        try:
            await asyncio.get_event_loop().run_in_executor(None, w.activate)
            await asyncio.sleep(0.4)
            return {"success": True, "action": "focus_window", "title": w.title}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def minimize(self, title_fragment: str) -> dict:
        """Minimize a window."""
        log_action("window", "minimize", title_fragment, dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}
        w = self.find_window(title_fragment)
        if not w:
            return {"success": False, "error": f"Window '{title_fragment}' not found."}
        try:
            await asyncio.get_event_loop().run_in_executor(None, w.minimize)
            return {"success": True, "action": "minimize"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def maximize(self, title_fragment: str) -> dict:
        """Maximize a window."""
        log_action("window", "maximize", title_fragment, dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}
        w = self.find_window(title_fragment)
        if not w:
            return {"success": False, "error": f"Window '{title_fragment}' not found."}
        try:
            await asyncio.get_event_loop().run_in_executor(None, w.maximize)
            return {"success": True, "action": "maximize"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def restore(self, title_fragment: str) -> dict:
        """Restore a minimized/maximized window to normal size."""
        log_action("window", "restore", title_fragment, dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}
        w = self.find_window(title_fragment)
        if not w:
            return {"success": False, "error": f"Window '{title_fragment}' not found."}
        try:
            await asyncio.get_event_loop().run_in_executor(None, w.restore)
            return {"success": True, "action": "restore"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def close_window(self, title_fragment: str) -> dict:
        """Close a window gracefully."""
        log_action("window", "close", title_fragment, dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}
        w = self.find_window(title_fragment)
        if not w:
            return {"success": False, "error": f"Window '{title_fragment}' not found."}
        try:
            await asyncio.get_event_loop().run_in_executor(None, w.close)
            return {"success": True, "action": "close_window"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def wait_for_window(
        self, title_fragment: str, timeout: float | None = None
    ) -> dict:
        """Wait until a window with given title appears (for app startup detection)."""
        deadline = asyncio.get_event_loop().time() + (timeout or automation_config.APP_STARTUP_TIMEOUT)
        while asyncio.get_event_loop().time() < deadline:
            if self.is_window_open(title_fragment):
                w = self.find_window(title_fragment)
                return {"success": True, "title": w.title if w else title_fragment}
            await asyncio.sleep(0.5)
        return {"success": False, "error": f"Window '{title_fragment}' did not appear within timeout."}

    async def switch_window(self, title_fragment: str) -> dict:
        """Focus and bring to front a specific window."""
        return await self.focus_window(title_fragment)

    async def move_window(self, title_fragment: str, x: int, y: int) -> dict:
        """Move a window to screen coordinates (x, y)."""
        log_action("window", "move", f"{title_fragment} -> ({x},{y})", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}
        w = self.find_window(title_fragment)
        if not w:
            return {"success": False, "error": f"Window '{title_fragment}' not found."}
        try:
            await asyncio.get_event_loop().run_in_executor(None, lambda: w.moveTo(x, y))
            return {"success": True, "action": "move_window"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def resize_window(self, title_fragment: str, width: int, height: int) -> dict:
        """Resize a window."""
        log_action("window", "resize", f"{title_fragment} -> {width}x{height}", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}
        w = self.find_window(title_fragment)
        if not w:
            return {"success": False, "error": f"Window '{title_fragment}' not found."}
        try:
            await asyncio.get_event_loop().run_in_executor(None, lambda: w.resizeTo(width, height))
            return {"success": True, "action": "resize_window"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Global singleton
window_controller = WindowController()

