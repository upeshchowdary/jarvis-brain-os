"""JARVIS Screen Observer — Wraps existing Vision + OCR modules.

Does NOT duplicate existing functionality.
Integrates directly with:
  - vision.screen_capture.ScreenCaptureEngine (screenshot)
  - vision.screen_analyzer.analyze_screen (Gemini vision)
  - vision.ocr.OCREngine (text extraction)
  - vision.environment.OCRTextItem (data model)
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from automation.config import automation_config
from automation.automation_logger import log_action

# Reuse existing vision modules
from vision.screen_capture import screen_capture_engine
from vision.ocr import OCREngine
from vision.environment import OCRTextItem, WindowInfo


@dataclass
class ScreenElement:
    """A detected interactive element on screen."""
    type: str          # button | input | text | link | menu | icon | error
    text: str
    x: int
    y: int
    width: int = 0
    height: int = 0
    confidence: float = 1.0

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class ScreenState:
    """Structured representation of the current screen state."""
    screenshot: Any = None            # PIL.Image
    active_app: str = ""
    window_title: str = ""
    ocr_text: str = ""                # Full extracted text
    ocr_items: List[OCRTextItem] = field(default_factory=list)
    vision_description: str = ""     # Gemini/LLM description
    elements: List[ScreenElement] = field(default_factory=list)
    raw_vision_result: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


class ScreenObserver:
    """
    Central screen observation engine.
    Combines screenshot capture, OCR, and AI vision into one interface.
    """

    def __init__(self) -> None:
        self._ocr_engine = OCREngine()
        self._last_state: Optional[ScreenState] = None

    # ── Screenshot ───────────────────────────────────────────────────

    def capture(self):
        """Capture the current full desktop screenshot. Returns PIL.Image."""
        try:
            screenshot = screen_capture_engine.capture_full_desktop()
            return screenshot
        except Exception as e:
            log_action("observer", "screenshot", result=f"ERROR: {e}", level="ERROR")
            return None

    def get_active_window_info(self) -> Optional[WindowInfo]:
        """Get active window info from existing screen capture engine."""
        try:
            return screen_capture_engine.get_active_window_info()
        except Exception:
            return None

    # ── OCR ──────────────────────────────────────────────────────────

    async def extract_text(self, image=None) -> Tuple[str, List[OCRTextItem]]:
        """Run OCR on a screenshot. Returns (full_text, items_with_positions)."""
        if image is None:
            image = self.capture()
        if image is None:
            return "", []
        try:
            items = await self._ocr_engine.extract_text_items(image)
            full_text = " ".join(item.text for item in items if item.text.strip())
            return full_text, items
        except Exception as e:
            log_action("observer", "ocr", result=f"ERROR: {e}", level="WARNING")
            return "", []

    # ── Vision AI ────────────────────────────────────────────────────

    async def analyze(self, query: str, image=None, session_id: str = "automation") -> ScreenState:
        """
        Full screen observation: capture → OCR → AI vision analysis.
        Returns a ScreenState with all data.
        """
        from vision.screen_analyzer import analyze_screen

        screenshot = image or self.capture()
        active_win = self.get_active_window_info()

        state = ScreenState(
            screenshot=screenshot,
            active_app=active_win.app_name if active_win else "",
            window_title=active_win.title if active_win else "",
        )

        if screenshot is None:
            state.error = "Screenshot capture failed"
            return state

        # Run OCR and Vision in parallel for speed
        ocr_task = asyncio.create_task(self.extract_text(screenshot))
        vision_task = asyncio.create_task(
            analyze_screen(
                image=screenshot,
                user_query=query,
                query_type="SCREEN_DESCRIPTION",
                session_id=session_id,
                window_title=state.window_title,
                app_name=state.active_app,
            )
        )

        # Wait with timeout
        try:
            ocr_result, vision_result = await asyncio.wait_for(
                asyncio.gather(ocr_task, vision_task),
                timeout=automation_config.VISION_TIMEOUT,
            )
            state.ocr_text, state.ocr_items = ocr_result
            state.vision_description = vision_result.get("description", "")
            state.raw_vision_result = vision_result
        except asyncio.TimeoutError:
            state.error = "Vision/OCR timed out"
            log_action("observer", "analyze", query, result="TIMEOUT", level="WARNING")

        self._last_state = state
        return state

    # ── Element Finding ──────────────────────────────────────────────

    async def find_text_location(
        self, search_text: str, image=None
    ) -> Optional[Tuple[int, int]]:
        """
        Find the screen coordinates of a specific text using OCR.
        Returns (center_x, center_y) or None if not found.
        """
        screenshot = image or self.capture()
        if screenshot is None:
            return None

        _, items = await self.extract_text(screenshot)
        search_lower = search_text.lower().strip()

        # Exact match first
        for item in items:
            if search_lower == item.text.lower().strip():
                cx = item.bounding_box.x + item.bounding_box.width // 2
                cy = item.bounding_box.y + item.bounding_box.height // 2
                return (cx, cy)

        # Partial / contains match
        for item in items:
            if search_lower in item.text.lower():
                cx = item.bounding_box.x + item.bounding_box.width // 2
                cy = item.bounding_box.y + item.bounding_box.height // 2
                return (cx, cy)

        return None

    async def find_element_by_description(
        self, description: str, image=None
    ) -> Optional[Tuple[int, int]]:
        """
        Use Vision AI to find an element described in natural language.
        Returns estimated (x, y) center coordinates.
        """
        query = f"Where is the '{description}' element? Give its approximate pixel coordinates."
        state = await self.analyze(query, image=image)

        # Try to extract coordinates from vision response
        import re
        text = state.vision_description or ""
        # Look for patterns like "(850, 620)" or "x=850, y=620" or "850x620"
        coord_match = re.search(r'\((\d+),\s*(\d+)\)|x[=:\s]+(\d+)[,\s]+y[=:\s]+(\d+)', text)
        if coord_match:
            groups = coord_match.groups()
            if groups[0] and groups[1]:
                return (int(groups[0]), int(groups[1]))
            elif groups[2] and groups[3]:
                return (int(groups[2]), int(groups[3]))

        return None

    async def wait_for_text(
        self,
        text: str,
        timeout: float | None = None,
        poll_interval: float = 0.5,
    ) -> bool:
        """Poll screen until text appears via OCR. Returns True when found."""
        deadline = asyncio.get_event_loop().time() + (timeout or automation_config.ELEMENT_SEARCH_TIMEOUT)
        while asyncio.get_event_loop().time() < deadline:
            loc = await self.find_text_location(text)
            if loc:
                return True
            await asyncio.sleep(poll_interval)
        return False

    async def wait_for_text_gone(
        self, text: str, timeout: float | None = None, poll_interval: float = 0.5
    ) -> bool:
        """Poll screen until text disappears. Returns True when gone."""
        deadline = asyncio.get_event_loop().time() + (timeout or automation_config.VERIFICATION_TIMEOUT)
        while asyncio.get_event_loop().time() < deadline:
            loc = await self.find_text_location(text)
            if not loc:
                return True
            await asyncio.sleep(poll_interval)
        return False

    @property
    def last_state(self) -> Optional[ScreenState]:
        return self._last_state


# Global singleton
screen_observer = ScreenObserver()
