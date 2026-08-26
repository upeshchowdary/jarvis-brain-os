"""Multi-Modal Perception Engine for JARVIS Automation Engine.

Combines signals across:
1. Native Windows UI Automation (pywinauto / Windows Accessibility)
2. Web DOM & Accessibility Tree (Playwright / CDP)
3. Window & Process Information (Win32 API)
4. OCR & Visual Perception (Bounding box localized text)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from automation.automation_logger import automation_logger


@dataclass
class UIAControlInfo:
    """Detailed metadata for a native Windows UI control."""
    name: str
    control_type: str  # Button, Edit, ListItem, Tab, MenuItem, Text, Window, etc.
    auto_id: str
    is_enabled: bool
    is_visible: bool
    bounding_rect: Dict[str, int]  # left, top, right, bottom, width, height
    center_point: Tuple[int, int]
    handle: int = 0


@dataclass
class DOMElementInfo:
    """Detailed metadata for a web element in the active browser page."""
    tag: str
    role: str
    name: str
    placeholder: str
    value: str
    href: str
    is_visible: bool
    selector: str
    bounding_rect: Dict[str, int]
    center_point: Tuple[int, int]


@dataclass
class ScreenState:
    """Unified snapshot of the current multi-modal computer perception."""
    timestamp: float = field(default_factory=time.time)
    active_window_title: str = ""
    active_process_name: str = ""
    active_url: str = ""
    is_browser: bool = False
    browser_type: str = ""  # "chrome", "edge", "playwright", "none"
    uia_controls: List[UIAControlInfo] = field(default_factory=list)
    dom_elements: List[DOMElementInfo] = field(default_factory=list)
    ocr_text_items: List[Dict[str, Any]] = field(default_factory=list)
    full_ocr_text: str = ""
    screenshot_image: Any = None


class PerceptionEngine:
    """Unified multi-modal perception engine."""

    def __init__(self) -> None:
        pass

    # ── 1. Window & Process State ─────────────────────────────────────

    def inspect_active_window(self) -> Dict[str, Any]:
        """Inspects active foreground window using Win32 API."""
        try:
            import win32gui
            import win32process
            import psutil

            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return {"hwnd": 0, "title": "", "process_name": "", "rect": {}}

            title = win32gui.GetWindowText(hwnd)
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            try:
                proc = psutil.Process(pid)
                pname = proc.name().lower()
            except Exception:
                pname = "unknown"

            rect = win32gui.GetWindowRect(hwnd)
            return {
                "hwnd": hwnd,
                "title": title,
                "process_name": pname,
                "rect": {
                    "left": rect[0],
                    "top": rect[1],
                    "right": rect[2],
                    "bottom": rect[3],
                    "width": rect[2] - rect[0],
                    "height": rect[3] - rect[1],
                },
            }
        except Exception as e:
            automation_logger.debug(f"[Perception] Active window inspection fallback: {e}")
            return {"hwnd": 0, "title": "", "process_name": "", "rect": {}}

    # ── 2. Windows UI Automation Tree ─────────────────────────────────

    def inspect_uia_controls(self, max_controls: int = 40) -> List[UIAControlInfo]:
        """
        Inspects native accessibility control tree of the active window.
        Extracts actionable controls (Buttons, Edits, ListItems, Tabs, Menus).
        """
        controls: List[UIAControlInfo] = []
        try:
            from pywinauto import Desktop  # type: ignore

            desktop = Desktop(backend="uia")
            active_win = desktop.top_window()
            if not active_win.exists(timeout=0.5):
                return []

            descendants = active_win.descendants()
            for elem in descendants[:max_controls * 2]:
                try:
                    c_type = elem.element_info.control_type or ""
                    # Filter for interactive controls
                    if c_type in ("Button", "Edit", "ListItem", "TabItem", "MenuItem", "CheckBox", "ComboBox", "Hyperlink", "Text"):
                        name = elem.element_info.name or ""
                        auto_id = elem.element_info.auto_id or ""
                        rect = elem.rectangle()
                        if rect.width() <= 0 or rect.height() <= 0:
                            continue

                        cx = (rect.left + rect.right) // 2
                        cy = (rect.top + rect.bottom) // 2

                        info = UIAControlInfo(
                            name=name,
                            control_type=c_type,
                            auto_id=auto_id,
                            is_enabled=elem.is_enabled() if hasattr(elem, "is_enabled") else True,
                            is_visible=elem.is_visible() if hasattr(elem, "is_visible") else True,
                            bounding_rect={
                                "left": rect.left,
                                "top": rect.top,
                                "right": rect.right,
                                "bottom": rect.bottom,
                                "width": rect.width(),
                                "height": rect.height(),
                            },
                            center_point=(cx, cy),
                            handle=elem.handle if hasattr(elem, "handle") else 0,
                        )
                        controls.append(info)
                        if len(controls) >= max_controls:
                            break
                except Exception:
                    continue
        except Exception as e:
            automation_logger.debug(f"[Perception] UIA inspection: {e}")

        return controls

    # ── 3. Visual & OCR Perception ────────────────────────────────────

    async def inspect_visual(self) -> Tuple[Any, str, List[Dict[str, Any]]]:
        """Captures screenshot and extracts OCR text with bounding boxes."""
        from automation.screen_observer import screen_observer
        screenshot = screen_observer.capture()
        if screenshot is None:
            return None, "", []

        full_text, items = await screen_observer.extract_text(screenshot)
        raw_items = []
        for it in items:
            raw_items.append({
                "text": it.text,
                "confidence": it.confidence,
                "x": it.bounding_box.x,
                "y": it.bounding_box.y,
                "width": it.bounding_box.width,
                "height": it.bounding_box.height,
                "center": (
                    it.bounding_box.x + it.bounding_box.width // 2,
                    it.bounding_box.y + it.bounding_box.height // 2,
                ),
            })
        return screenshot, full_text, raw_items

    # ── 4. Unified Perception Snapshot ────────────────────────────────

    async def capture_state(self, include_visual: bool = True) -> ScreenState:
        """Takes a full multi-modal snapshot of the computer environment."""
        win_info = self.inspect_active_window()
        pname = win_info.get("process_name", "")
        title = win_info.get("title", "")
        is_browser = pname in ("chrome.exe", "msedge.exe", "firefox.exe", "brave.exe")

        state = ScreenState(
            active_window_title=title,
            active_process_name=pname,
            is_browser=is_browser,
            browser_type="chrome" if "chrome" in pname else ("edge" if "edge" in pname else ""),
        )

        # UIA controls for native/browser windows
        loop = asyncio.get_event_loop()
        state.uia_controls = await loop.run_in_executor(None, self.inspect_uia_controls, 30)

        # Visual & OCR
        if include_visual:
            img, text, ocr_items = await self.inspect_visual()
            state.screenshot_image = img
            state.full_ocr_text = text
            state.ocr_text_items = ocr_items

        return state


perception_engine = PerceptionEngine()
