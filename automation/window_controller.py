"""JARVIS Window Controller — Application window management via Windows API.

Uses pygetwindow + win32api for window listing, focus, resize, and state control.
Falls back to graceful errors when windows are not found.
"""

import asyncio
import re
import time
from typing import Dict, List, Optional, Set, Tuple

from automation.automation_logger import log_action

try:
    import pygetwindow as gw  # type: ignore
    HAS_GW = True
except ImportError:
    HAS_GW = False

# Module-level so tests can drive readiness on a fake clock.
_clock = time.monotonic
_sleep = asyncio.sleep

# Never taken for another app's window by its title (a tab can be called anything).
_BROWSER_PROCESSES = {"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe", "vivaldi.exe"}


def _process_name(hwnd: int) -> str:
    """Lower-case exe name of the process that owns ``hwnd`` ("" when unknown)."""
    try:
        import psutil  # type: ignore
        import win32process  # type: ignore

        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        return str(psutil.Process(int(pid)).name()).lower()
    except Exception:
        return ""


def _title_names(title: str, name: str) -> bool:
    """``name`` appears in ``title`` as a whole name: "Notes - Notepad" yes, "new 1 - Notepad++" no."""
    if not name.strip():
        return False
    return re.search(rf"(?<![\w+]){re.escape(name.strip())}(?![\w+])", title, re.IGNORECASE) is not None

# Reuse existing WindowInfo model from vision layer


class WindowController:
    """Manages application windows: find, focus, minimize, maximize, close, list."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run  # global dry-run is enforced by the gateway

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

    def window_snapshot(self) -> Dict[int, Tuple[str, str]]:
        """``{hwnd: (title, process exe name)}`` of every titled top-level window ({} when unknown)."""
        if not HAS_GW:
            return {}
        try:
            snap: Dict[int, Tuple[str, str]] = {}
            for w in gw.getAllWindows():
                hwnd = getattr(w, "_hWnd", None)
                if isinstance(hwnd, int) and isinstance(w.title, str) and w.title.strip():
                    snap[hwnd] = (w.title, _process_name(hwnd))
            return snap
        except Exception:
            return {}

    def foreground_window(self) -> Tuple[Optional[int], str]:
        """``(hwnd, title)`` of the foreground window; ``(None, "")`` when unknown."""
        try:
            import win32gui  # type: ignore

            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)
            if isinstance(hwnd, int):
                return hwnd, title if isinstance(title, str) else ""
        except Exception:
            pass
        return None, ""

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
        deadline = asyncio.get_event_loop().time() + (timeout or 2.5)
        while asyncio.get_event_loop().time() < deadline:
            if self.is_window_open(title_fragment):
                w = self.find_window(title_fragment)
                return {"success": True, "title": w.title if w else title_fragment}
            await asyncio.sleep(0.05)
        return {"success": False, "error": f"Window '{title_fragment}' did not appear within timeout."}

    async def wait_for_app_window(
        self,
        name: str,
        before: Dict[int, Tuple[str, str]],
        before_foreground: Tuple[Optional[int], str],
        timeout: float = 3.0,
        settle: float = 0.75,
        poll: float = 0.1,
        processes: Optional[Set[str]] = None,
    ) -> dict:
        """After a launch, wait (bounded by ``timeout``) for the window the app ended up in.

        ``before`` / ``before_foreground`` are the windows (``window_snapshot``) and the foreground window just
        before the launch. A window is the app's when its owning process is one of ``processes`` (the launched
        exe); only when that isn't known (Store / shell / protocol launches) is its title the hint: ``name`` as a
        whole name, never a browser window. So a popup, a UAC prompt, Notepad++ or a browser tab mentioning
        "notepad" is never taken for Notepad. Single-instance / tabbed apps (Windows 11 Notepad, Terminal,
        browsers, Explorer) reuse a window, so a new window may never appear. Ready, with ``kind``:

        - ``new_window``: an app window that didn't exist before;
        - ``new_tab``: an existing app window is in front and its title changed (a new tab / document);
        - ``existing``: an existing app window is in front, title unchanged for ``settle`` seconds
          (the old window was just brought to the front: it may hold the user's own unsaved document).
        """
        procs = {p.lower() for p in (processes or ()) if p}

        def is_app(title: str, proc: str) -> bool:
            if procs:
                return proc in procs
            return _title_names(title, name) and proc not in _BROWSER_PROCESSES

        deadline = _clock() + timeout
        seen_existing: Optional[float] = None
        while True:
            now_windows = await asyncio.to_thread(self.window_snapshot)
            fg_hwnd, fg_title = await asyncio.to_thread(self.foreground_window)
            app_windows = {h: t for h, (t, proc) in now_windows.items() if is_app(t, proc)}
            new = [h for h in app_windows if h not in before]
            if new:
                hwnd = fg_hwnd if fg_hwnd in new else new[0]
                return {"success": True, "kind": "new_window", "hwnd": hwnd, "title": app_windows[hwnd]}
            if fg_hwnd is not None and fg_hwnd in app_windows:
                old_title = before[fg_hwnd][0] if fg_hwnd in before else None
                if old_title is not None and fg_title != old_title:
                    return {"success": True, "kind": "new_tab", "hwnd": fg_hwnd, "title": fg_title}
                if seen_existing is None:
                    seen_existing = _clock()
                elif _clock() - seen_existing >= settle:
                    return {"success": True, "kind": "existing", "hwnd": fg_hwnd, "title": fg_title}
            else:
                seen_existing = None
            if _clock() >= deadline:
                return {"success": False, "error": f"No window for '{name}' came to the front within {timeout:g}s."}
            await _sleep(poll)

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

