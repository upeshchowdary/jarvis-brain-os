"""Hierarchical Action Router for JARVIS Automation Engine.

Implements the 6-Level Action Hierarchy:
Level 1: Direct API (open app, read file) — no shell, ever
Level 2: DOM & Web Accessibility Locators (Playwright / Semantic Selectors)
Level 3: Windows UI Automation (pywinauto native controls by Name/Id/Type)
Level 4: Keyboard Navigation & Fast Clipboard Text Injection
Level 5: Precision Mouse Interaction (Element Bounding-Box Centered)
Level 6: Multimodal Computer Vision (Grounded fallback perception)
"""

from __future__ import annotations

import asyncio
import enum
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

from automation.automation_logger import automation_logger


class ActionLevel(enum.Enum):
    LEVEL_1_DIRECT_API = "L1_DIRECT_API"
    LEVEL_2_DOM_WEB = "L2_DOM_WEB"
    LEVEL_3_WINDOWS_UIA = "L3_WINDOWS_UIA"
    LEVEL_4_KEYBOARD = "L4_KEYBOARD"
    LEVEL_5_MOUSE = "L5_MOUSE"
    LEVEL_6_VISION = "L6_VISION"


@dataclass
class ActionSpec:
    """Specification of an action to be routed and executed."""
    level: ActionLevel
    action_type: str  # "click", "type", "press", "navigate", "open_app", "scroll", "hover", "select", "hotkey"
    target: str = ""
    value: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 5.0


@dataclass
class ActionExecutionResult:
    """Outcome of an action execution."""
    success: bool
    level_used: ActionLevel
    action_type: str
    target: str
    elapsed_ms: float
    output_data: Any = None
    error: Optional[str] = None


class ActionRouter:
    """Executes actions by selecting the most reliable, fastest mechanism in the hierarchy."""

    def __init__(self) -> None:
        self._playwright_page = None
        self._playwright_browser = None

    # ── Level 1: Direct API / CLI / Filesystem ─────────────────────────

    async def execute_l1_direct(self, action: ActionSpec) -> ActionExecutionResult:
        """Level 1: Direct API, CLI, or filesystem operations."""
        start = time.perf_counter()
        atype = action.action_type.lower()
        target = action.target

        try:
            if atype == "open_app":
                from automation.application_controller import application_controller
                res = await application_controller.open_app(target)
                elapsed = (time.perf_counter() - start) * 1000
                return ActionExecutionResult(
                    success=res.get("success", False),
                    level_used=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type=atype,
                    target=target,
                    elapsed_ms=elapsed,
                    output_data=res,
                )

            elif atype == "read_file":
                expanded = os.path.expandvars(os.path.expanduser(target))
                with open(expanded, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                elapsed = (time.perf_counter() - start) * 1000
                return ActionExecutionResult(
                    success=True,
                    level_used=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type=atype,
                    target=target,
                    elapsed_ms=elapsed,
                    output_data=content,
                )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return ActionExecutionResult(
                success=False,
                level_used=ActionLevel.LEVEL_1_DIRECT_API,
                action_type=atype,
                target=target,
                elapsed_ms=elapsed,
                error=str(e),
            )

        return ActionExecutionResult(
            success=False,
            level_used=ActionLevel.LEVEL_1_DIRECT_API,
            action_type=atype,
            target=target,
            elapsed_ms=0,
            error=f"Unsupported L1 action: {atype}",
        )


    # ── Level 2: DOM & Web Accessibility (Playwright) ─────────────────

    async def get_or_attach_browser_page(self):
        """Attaches to existing Playwright browser or launches Chromium."""
        if self._playwright_page is not None and not self._playwright_page.is_closed():
            return self._playwright_page

        try:
            from playwright.async_api import async_playwright
            p = await async_playwright().start()
            # Try connecting over CDP to existing Chrome instance
            try:
                browser = await p.chromium.connect_over_cdp("http://localhost:9222", timeout=1500)
                if browser.contexts:
                    pages = browser.contexts[0].pages
                    if pages:
                        self._playwright_page = pages[0]
                        self._playwright_browser = browser
                        return self._playwright_page
            except Exception:
                pass

            # Launch local browser instance if CDP not active
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            self._playwright_page = page
            self._playwright_browser = browser
            return page
        except Exception as e:
            automation_logger.debug(f"[ActionRouter] Playwright browser attach: {e}")
            return None

    async def execute_l2_web(self, action: ActionSpec) -> ActionExecutionResult:
        """Level 2: Web DOM & browser actions."""
        start = time.perf_counter()
        atype = action.action_type.lower()
        target = action.target
        value = action.value

        if atype == "navigate":
            from automation.web_utils import search_url, site_url

            url = site_url(target) or search_url(target)

            # 1. OS default-browser handler (works even when no browser is open), then the active window
            import webbrowser
            from automation.browser_controller import browser_controller
            try:
                opened = await asyncio.to_thread(webbrowser.open, url, 2)
                if opened:
                    await asyncio.sleep(1.2)  # let the page start loading before the next step
                nav_res = {"success": True} if opened else await browser_controller.navigate(url)
                if nav_res.get("success"):
                    elapsed = (time.perf_counter() - start) * 1000
                    return ActionExecutionResult(
                        success=True,
                        level_used=ActionLevel.LEVEL_2_DOM_WEB,
                        action_type=atype,
                        target=url,
                        elapsed_ms=elapsed,
                    )
            except Exception as e:
                automation_logger.warning(f"[ActionRouter] Desktop browser navigate error: {e}")

            # 2. Fallback to Playwright if available
            try:
                page = await self.get_or_attach_browser_page()
                if page is not None:
                    await page.goto(url, wait_until="domcontentloaded", timeout=10000)
                    elapsed = (time.perf_counter() - start) * 1000
                    return ActionExecutionResult(
                        success=True,
                        level_used=ActionLevel.LEVEL_2_DOM_WEB,
                        action_type=atype,
                        target=url,
                        elapsed_ms=elapsed,
                    )
            except Exception as e:
                automation_logger.warning(f"[ActionRouter] Playwright fallback error: {e}")

            elapsed = (time.perf_counter() - start) * 1000
            return ActionExecutionResult(
                success=False,
                level_used=ActionLevel.LEVEL_2_DOM_WEB,
                action_type=atype,
                target=url,
                elapsed_ms=elapsed,
                error="Could not open the browser",
            )

        try:
            page = await self.get_or_attach_browser_page()
            if page is None:
                raise RuntimeError("No active Playwright page available for L2 web execution")

            elif atype == "click":
                # Try semantic locators in order: role -> text -> placeholder -> selector
                clicked = False
                for loc_expr in [
                    f'button:has-text("{target}")',
                    f'a:has-text("{target}")',
                    f'[aria-label*="{target}" i]',
                    f'[title*="{target}" i]',
                    f'text="{target}"',
                    target,
                ]:
                    try:
                        loc = page.locator(loc_expr).first
                        if await loc.is_visible(timeout=1000):
                            await loc.click(timeout=3000)
                            clicked = True
                            break
                    except Exception:
                        continue

                elapsed = (time.perf_counter() - start) * 1000
                return ActionExecutionResult(
                    success=clicked,
                    level_used=ActionLevel.LEVEL_2_DOM_WEB,
                    action_type=atype,
                    target=target,
                    elapsed_ms=elapsed,
                    error=None if clicked else f"Web element '{target}' not clickable via DOM",
                )

            elif atype == "type" or atype == "fill":
                filled = False
                for loc_expr in [
                    f'input[placeholder*="{target}" i]',
                    f'input[aria-label*="{target}" i]',
                    f'[data-testid*="{target}" i]',
                    target,
                ]:
                    try:
                        loc = page.locator(loc_expr).first
                        if await loc.is_visible(timeout=1000):
                            await loc.fill(value, timeout=3000)
                            filled = True
                            break
                    except Exception:
                        continue

                if not filled:
                    # Fallback: type into active element
                    await page.keyboard.type(value)
                    filled = True

                elapsed = (time.perf_counter() - start) * 1000
                return ActionExecutionResult(
                    success=filled,
                    level_used=ActionLevel.LEVEL_2_DOM_WEB,
                    action_type=atype,
                    target=target,
                    elapsed_ms=elapsed,
                )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return ActionExecutionResult(
                success=False,
                level_used=ActionLevel.LEVEL_2_DOM_WEB,
                action_type=atype,
                target=target,
                elapsed_ms=elapsed,
                error=str(e),
            )

        return ActionExecutionResult(
            success=False,
            level_used=ActionLevel.LEVEL_2_DOM_WEB,
            action_type=atype,
            target=target,
            elapsed_ms=0,
            error=f"Unsupported L2 action: {atype}",
        )

    # ── Level 3: Windows UI Automation ────────────────────────────────

    async def execute_l3_uia(self, action: ActionSpec) -> ActionExecutionResult:
        """Level 3: Windows UI Automation native controls."""
        start = time.perf_counter()
        atype = action.action_type.lower()
        target = action.target
        value = action.value

        def _sync_uia_exec():
            from pywinauto import Desktop
            from pywinauto.keyboard import send_keys
            import time as _t

            try:
                desktop = Desktop(backend="uia")
                active_win = desktop.top_window()
                if not active_win.exists(timeout=1):
                    return False, "No active window found for UIA"
            except Exception as e:
                return False, f"UIA Desktop init failed: {e}"

            if atype == "click":
                # Search by title, name, or auto_id
                ctrl = None
                for c_type in ("Button", "ListItem", "TabItem", "MenuItem", "Hyperlink", "Text", "Edit"):
                    try:
                        ctrl = active_win.child_window(title_re=f".*{target}.*", control_type=c_type)
                        if ctrl.exists(timeout=0.5):
                            ctrl.click_input()
                            return True, f"Clicked UIA control: {c_type} '{target}'"
                    except Exception:
                        continue

                return False, f"UIA control '{target}' not found in active window"

            elif atype == "type" or atype == "fill":
                ctrl = None
                try:
                    ctrl = active_win.child_window(title_re=f".*{target}.*", control_type="Edit")
                    if ctrl.exists(timeout=0.5):
                        ctrl.click_input()
                        _t.sleep(0.1)
                        send_keys("^a{BACKSPACE}")
                        ctrl.type_keys(value, with_spaces=True)
                        return True, f"Typed into UIA edit '{target}'"
                except Exception:
                    pass

                # Fallback: send keys directly to window
                send_keys(value.replace("{", "{{").replace("}", "}}"), with_spaces=True)
                return True, "Typed directly via UIA keyboard"

            return False, f"Unsupported L3 UIA action: {atype}"

        loop = asyncio.get_event_loop()
        success, msg = await loop.run_in_executor(None, _sync_uia_exec)
        elapsed = (time.perf_counter() - start) * 1000

        return ActionExecutionResult(
            success=success,
            level_used=ActionLevel.LEVEL_3_WINDOWS_UIA,
            action_type=atype,
            target=target,
            elapsed_ms=elapsed,
            error=None if success else msg,
            output_data=msg if success else None,
        )


    # ── Level 4: Keyboard Navigation & Clipboard ──────────────────────

    async def execute_l4_keyboard(self, action: ActionSpec) -> ActionExecutionResult:
        """Level 4: Fast keyboard actions and clipboard paste text injection."""
        start = time.perf_counter()
        atype = action.action_type.lower()
        value = action.value or action.target

        from automation.keyboard_controller import keyboard_controller

        try:
            if atype == "type":
                # Fast injection: long text uses clipboard paste, short uses type_text
                if len(value) > 20:
                    import pyperclip
                    saved = None
                    try:
                        saved = pyperclip.paste()
                    except Exception:
                        pass
                    pyperclip.copy(value)
                    await keyboard_controller.hotkey("ctrl", "v")
                    await asyncio.sleep(0.15)
                    if saved is not None:
                        pyperclip.copy(saved)
                else:
                    await keyboard_controller.type_text(value)

                elapsed = (time.perf_counter() - start) * 1000
                return ActionExecutionResult(
                    success=True,
                    level_used=ActionLevel.LEVEL_4_KEYBOARD,
                    action_type=atype,
                    target=action.target,
                    elapsed_ms=elapsed,
                )

            elif atype == "press":
                await keyboard_controller.press(value)
                elapsed = (time.perf_counter() - start) * 1000
                return ActionExecutionResult(
                    success=True,
                    level_used=ActionLevel.LEVEL_4_KEYBOARD,
                    action_type=atype,
                    target=action.target,
                    elapsed_ms=elapsed,
                )

            elif atype == "hotkey":
                keys = [k.strip() for k in value.split("+")]
                await keyboard_controller.hotkey(*keys)
                elapsed = (time.perf_counter() - start) * 1000
                return ActionExecutionResult(
                    success=True,
                    level_used=ActionLevel.LEVEL_4_KEYBOARD,
                    action_type=atype,
                    target=action.target,
                    elapsed_ms=elapsed,
                )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return ActionExecutionResult(
                success=False,
                level_used=ActionLevel.LEVEL_4_KEYBOARD,
                action_type=atype,
                target=action.target,
                elapsed_ms=elapsed,
                error=str(e),
            )

        return ActionExecutionResult(
            success=False,
            level_used=ActionLevel.LEVEL_4_KEYBOARD,
            action_type=atype,
            target=action.target,
            elapsed_ms=0,
            error=f"Unsupported L4 keyboard action: {atype}",
        )

    # ── Level 5: Precision Mouse ──────────────────────────────────────

    async def execute_l5_mouse(self, action: ActionSpec) -> ActionExecutionResult:
        """Level 5: Precision mouse clicking on verified element bounding box centers."""
        start = time.perf_counter()
        atype = action.action_type.lower()
        target = action.target

        from automation.mouse_controller import mouse_controller
        from automation.engine.perception import perception_engine

        try:
            # 1. Check if coordinates explicitly provided
            if "x" in action.parameters and "y" in action.parameters:
                x, y = int(action.parameters["x"]), int(action.parameters["y"])
                await mouse_controller.click(x, y)
                elapsed = (time.perf_counter() - start) * 1000
                return ActionExecutionResult(
                    success=True,
                    level_used=ActionLevel.LEVEL_5_MOUSE,
                    action_type=atype,
                    target=target,
                    elapsed_ms=elapsed,
                )

            # 2. Find target center from OCR bounding boxes
            _, _, ocr_items = await perception_engine.inspect_visual()
            matched_item = None
            target_low = target.lower()

            for item in ocr_items:
                if target_low in item["text"].lower():
                    matched_item = item
                    break

            if matched_item:
                cx, cy = matched_item["center"]
                await mouse_controller.click(cx, cy)
                elapsed = (time.perf_counter() - start) * 1000
                return ActionExecutionResult(
                    success=True,
                    level_used=ActionLevel.LEVEL_5_MOUSE,
                    action_type=atype,
                    target=target,
                    elapsed_ms=elapsed,
                    output_data={"click_coords": (cx, cy)},
                )

            # 3. Fallback: check UIA controls
            controls = perception_engine.inspect_uia_controls()
            for c in controls:
                if target_low in c.name.lower():
                    cx, cy = c.center_point
                    await mouse_controller.click(cx, cy)
                    elapsed = (time.perf_counter() - start) * 1000
                    return ActionExecutionResult(
                        success=True,
                        level_used=ActionLevel.LEVEL_5_MOUSE,
                        action_type=atype,
                        target=target,
                        elapsed_ms=elapsed,
                        output_data={"click_coords": (cx, cy)},
                    )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return ActionExecutionResult(
                success=False,
                level_used=ActionLevel.LEVEL_5_MOUSE,
                action_type=atype,
                target=target,
                elapsed_ms=elapsed,
                error=str(e),
            )

        elapsed = (time.perf_counter() - start) * 1000
        return ActionExecutionResult(
            success=False,
            level_used=ActionLevel.LEVEL_5_MOUSE,
            action_type=atype,
            target=target,
            elapsed_ms=elapsed,
            error=f"Could not locate visual target '{target}' for mouse click",
        )

    # ── Level 6: Vision Perception ────────────────────────────────────

    async def execute_l6_vision(self, action: ActionSpec) -> ActionExecutionResult:
        """Level 6: Vision-guided grounded target discovery and action."""
        start = time.perf_counter()
        atype = action.action_type.lower()
        target = action.target

        from automation.screen_observer import screen_observer
        from automation.mouse_controller import mouse_controller
        from vision.screen_analyzer import screen_analyzer

        try:
            screenshot = screen_observer.capture()
            if screenshot is None:
                raise RuntimeError("Failed to capture screenshot for L6 vision")

            analysis = await screen_analyzer.find_element(
                screenshot=screenshot,
                element_description=target,
            )

            if analysis and analysis.get("found") and "bounding_box" in analysis:
                bbox = analysis["bounding_box"]
                cx = bbox["x"] + bbox["width"] // 2
                cy = bbox["y"] + bbox["height"] // 2
                await mouse_controller.click(cx, cy)
                elapsed = (time.perf_counter() - start) * 1000
                return ActionExecutionResult(
                    success=True,
                    level_used=ActionLevel.LEVEL_6_VISION,
                    action_type=atype,
                    target=target,
                    elapsed_ms=elapsed,
                    output_data={"coords": (cx, cy)},
                )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return ActionExecutionResult(
                success=False,
                level_used=ActionLevel.LEVEL_6_VISION,
                action_type=atype,
                target=target,
                elapsed_ms=elapsed,
                error=str(e),
            )

        elapsed = (time.perf_counter() - start) * 1000
        return ActionExecutionResult(
            success=False,
            level_used=ActionLevel.LEVEL_6_VISION,
            action_type=atype,
            target=target,
            elapsed_ms=elapsed,
            error=f"Vision model could not locate element: '{target}'",
        )

    # ── Master Execution Dispatcher ───────────────────────────────────

    async def dispatch(self, action: ActionSpec) -> ActionExecutionResult:
        """Dispatches action to the requested hierarchy level."""
        if action.level == ActionLevel.LEVEL_1_DIRECT_API:
            return await self.execute_l1_direct(action)
        elif action.level == ActionLevel.LEVEL_2_DOM_WEB:
            return await self.execute_l2_web(action)
        elif action.level == ActionLevel.LEVEL_3_WINDOWS_UIA:
            return await self.execute_l3_uia(action)
        elif action.level == ActionLevel.LEVEL_4_KEYBOARD:
            return await self.execute_l4_keyboard(action)
        elif action.level == ActionLevel.LEVEL_5_MOUSE:
            return await self.execute_l5_mouse(action)
        elif action.level == ActionLevel.LEVEL_6_VISION:
            return await self.execute_l6_vision(action)

        return ActionExecutionResult(
            success=False,
            level_used=action.level,
            action_type=action.action_type,
            target=action.target,
            elapsed_ms=0,
            error=f"Unknown action level: {action.level}",
        )


action_router = ActionRouter()
