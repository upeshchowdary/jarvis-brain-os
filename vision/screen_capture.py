"""JARVIS Screen Capture Engine.

Provides high-performance capturing of full desktop screens, multi-monitor setups,
specific application windows, bounding regions, active window metadata, and cursor position.
"""

import sys
import ctypes
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

try:
    from PIL import Image, ImageGrab
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from vision.environment import (
    BoundingBox,
    CursorState,
    MonitorInfo,
    WindowInfo,
)


class ScreenCaptureEngine:
    """Engine responsible for capturing screen displays, active windows, and cursor telemetry."""

    def __init__(self) -> None:
        self.is_windows = sys.platform == "win32"

    def get_cursor_position(self) -> CursorState:
        """Returns current mouse cursor coordinates (x, y) and state."""
        x, y = 0, 0
        if self.is_windows:
            try:
                class POINT(ctypes.Structure):
                    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
                pt = POINT()
                if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
                    x, y = pt.x, pt.y
            except Exception as e:
                logger.debug(f"Failed to get cursor pos via Windows API: {e}")
        return CursorState(x=x, y=y, visible=True, cursor_type="arrow")

    def get_monitors_info(self) -> List[MonitorInfo]:
        """Returns metadata for all available monitors."""
        monitors = []
        if HAS_PIL:
            try:
                # Primary monitor screen size
                img = ImageGrab.grab()
                w, h = img.size
                monitors.append(
                    MonitorInfo(
                        index=0,
                        name="Primary Display",
                        resolution=(w, h),
                        scaling=1.0,
                        is_primary=True,
                    )
                )
            except Exception as e:
                logger.debug(f"Primary monitor query fallback: {e}")

        if not monitors:
            monitors.append(
                MonitorInfo(
                    index=0,
                    name="Default Display",
                    resolution=(1920, 1080),
                    scaling=1.0,
                    is_primary=True,
                )
            )
        return monitors

    def get_active_window_info(self) -> WindowInfo:
        """Returns WindowInfo for the currently focused active window."""
        if self.is_windows:
            try:
                user32 = ctypes.windll.user32
                hwnd = user32.GetForegroundWindow()
                if hwnd:
                    length = user32.GetWindowTextLengthW(hwnd)
                    title_buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, title_buf, length + 1)
                    title = title_buf.value.strip()

                    # Get Window Rect [left, top, right, bottom]
                    class RECT(ctypes.Structure):
                        _fields_ = [
                            ("left", ctypes.c_long),
                            ("top", ctypes.c_long),
                            ("right", ctypes.c_long),
                            ("bottom", ctypes.c_long),
                        ]
                    rect = RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    w = max(0, rect.right - rect.left)
                    h = max(0, rect.bottom - rect.top)

                    # Extract app process name
                    pid = ctypes.c_ulong()
                    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                    app_name = self._get_process_name_by_pid(pid.value) or title

                    return WindowInfo(
                        window_id=int(hwnd),
                        title=title or "Active Desktop",
                        app_name=app_name,
                        is_active=True,
                        bounds=BoundingBox(x=rect.left, y=rect.top, width=w, height=h),
                    )
            except Exception as e:
                logger.debug(f"Failed to get active window info: {e}")

        return WindowInfo(
            window_id=0,
            title="Active Desktop",
            app_name="Desktop Environment",
            is_active=True,
            bounds=BoundingBox(x=0, y=0, width=1920, height=1080),
        )

    def get_all_open_windows(self) -> List[WindowInfo]:
        """Lists all open visible desktop windows."""
        windows = []
        if self.is_windows:
            try:
                user32 = ctypes.windll.user32
                from ctypes import wintypes

                active_hwnd = user32.GetForegroundWindow()

                def enum_cb(hwnd, lparam):
                    if user32.IsWindowVisible(hwnd):
                        length = user32.GetWindowTextLengthW(hwnd)
                        if length > 0:
                            buf = ctypes.create_unicode_buffer(length + 1)
                            user32.GetWindowTextW(hwnd, buf, length + 1)
                            t = buf.value.strip()
                            if t and t not in ("Program Manager", "Settings", "Default IME"):
                                class RECT(ctypes.Structure):
                                    _fields_ = [
                                        ("left", ctypes.c_long),
                                        ("top", ctypes.c_long),
                                        ("right", ctypes.c_long),
                                        ("bottom", ctypes.c_long),
                                    ]
                                rect = RECT()
                                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                                w = max(0, rect.right - rect.left)
                                h = max(0, rect.bottom - rect.top)

                                windows.append(
                                    WindowInfo(
                                        window_id=int(hwnd),
                                        title=t,
                                        app_name=t.split(" - ")[-1] if " - " in t else t,
                                        is_active=(hwnd == active_hwnd),
                                        bounds=BoundingBox(x=rect.left, y=rect.top, width=w, height=h),
                                    )
                                )
                    return True

                WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
                user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
            except Exception as e:
                logger.debug(f"Failed to enumerate windows: {e}")

        if not windows:
            windows.append(self.get_active_window_info())

        return windows

    def capture_full_desktop(self) -> Any:
        """Captures entire desktop display as a PIL Image."""
        # Use fast_capture as primary (DXGI/mss)
        try:
            from vision.fast_capture import capture_full_screen
            img = capture_full_screen()
            if img is not None:
                return img
        except Exception as e:
            logger.debug(f"fast_capture fallback: {e}")

        if not HAS_PIL:
            logger.error("PIL ImageGrab is not available.")
            return None
        try:
            return ImageGrab.grab(all_screens=True)
        except Exception as e:
            logger.warning(f"Full desktop capture fallback: {e}")
            return Image.new("RGB", (1920, 1080), color=(30, 30, 35))

    def capture_region(self, bbox: BoundingBox) -> Any:
        """Captures a specific screen bounding box region."""
        if not HAS_PIL:
            return None
        box_tuple = (bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height)
        try:
            return ImageGrab.grab(bbox=box_tuple)
        except Exception as e:
            logger.warning(f"Region capture fallback: {e}")
            return Image.new("RGB", (bbox.width, bbox.height), color=(30, 30, 35))

    def capture_window(self, window_title_or_id: Any) -> Tuple[Optional[Any], Optional[WindowInfo]]:
        """Captures frame of a specific window by title or handle ID."""
        target_win = None
        all_wins = self.get_all_open_windows()

        for win in all_wins:
            if isinstance(window_title_or_id, int) and win.window_id == window_title_or_id:
                target_win = win
                break
            elif isinstance(window_title_or_id, str) and window_title_or_id.lower() in win.title.lower():
                target_win = win
                break

        if target_win:
            img = self.capture_region(target_win.bounds)
            return img, target_win

        # Fallback to active window
        active_win = self.get_active_window_info()
        return self.capture_region(active_win.bounds), active_win

    def _get_process_name_by_pid(self, pid: int) -> str:
        """Attempts process name lookup for a process ID."""
        if not pid:
            return ""
        try:
            import psutil
            proc = psutil.Process(pid)
            return proc.name()
        except Exception:
            return ""


# Global singleton instance
screen_capture_engine = ScreenCaptureEngine()
