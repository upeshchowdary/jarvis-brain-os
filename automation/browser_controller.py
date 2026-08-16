"""JARVIS Browser Controller — Screen-based browser automation.

Uses keyboard shortcuts + Vision as primary strategy (works with any browser).
No Selenium/Playwright dependency required.
Playwright/Selenium can be plugged in later as a drop-in enhancement.
"""

import asyncio
from typing import Optional

from automation.config import automation_config
from automation.automation_logger import log_action
from automation.keyboard_controller import keyboard_controller
from automation.mouse_controller import mouse_controller
from automation.application_controller import application_controller
from automation.screen_observer import screen_observer
from automation.window_controller import window_controller


class BrowserController:
    """Controls browser via keyboard shortcuts and Vision-guided interaction."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run or automation_config.DRY_RUN
        self._default_browser = "chrome"

    # ── App Lifecycle ────────────────────────────────────────────────

    async def open(self, browser: str = "chrome") -> dict:
        """Open a browser application."""
        self._default_browser = browser.lower()
        log_action("browser", "open", browser, dry_run=self.dry_run)
        result = await application_controller.open_app(browser)
        if result["success"]:
            await asyncio.sleep(1.0)  # Let browser fully initialize
        return result

    async def ensure_open(self) -> dict:
        """Open browser if not already open."""
        browser = self._default_browser
        hints = {"chrome": "Chrome", "edge": "Edge", "firefox": "Firefox", "brave": "Brave"}
        hint = hints.get(browser, browser)
        if not window_controller.is_window_open(hint):
            return await self.open(browser)
        await window_controller.focus_window(hint)
        return {"success": True, "action": "ensure_open", "browser": browser}

    # ── Navigation ───────────────────────────────────────────────────

    async def navigate(self, url: str) -> dict:
        """Navigate to a URL by focusing address bar and typing."""
        log_action("browser", "navigate", url, dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "action": "navigate", "url": url}

        await self.ensure_open()
        # Ctrl+L focuses address bar in all major browsers
        await keyboard_controller.hotkey("ctrl", "l")
        await asyncio.sleep(0.3)
        await keyboard_controller.select_all()
        await keyboard_controller.type_text(url)
        await keyboard_controller.enter()
        # Wait for page to start loading
        await asyncio.sleep(1.5)
        return {"success": True, "action": "navigate", "url": url}

    async def new_tab(self) -> dict:
        """Open a new browser tab."""
        log_action("browser", "new_tab", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}
        await self.ensure_open()
        return await keyboard_controller.hotkey("ctrl", "t")

    async def close_tab(self) -> dict:
        """Close the current browser tab."""
        log_action("browser", "close_tab", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}
        return await keyboard_controller.hotkey("ctrl", "w")

    async def refresh(self) -> dict:
        """Refresh the current page."""
        log_action("browser", "refresh", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}
        return await keyboard_controller.hotkey("ctrl", "r")

    async def back(self) -> dict:
        """Go back in browser history."""
        log_action("browser", "back", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}
        return await keyboard_controller.hotkey("alt", "left")

    async def forward(self) -> dict:
        """Go forward in browser history."""
        log_action("browser", "forward", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}
        return await keyboard_controller.hotkey("alt", "right")

    async def switch_tab(self, direction: str = "next") -> dict:
        """Switch to next or previous tab."""
        log_action("browser", f"switch_tab ({direction})", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}
        if direction == "next":
            return await keyboard_controller.hotkey("ctrl", "tab")
        else:
            return await keyboard_controller.hotkey("ctrl", "shift", "tab")

    # ── Search ───────────────────────────────────────────────────────

    async def search(self, query: str, engine: str = "google") -> dict:
        """Open a search engine and search for a query."""
        log_action("browser", "search", query, dry_run=self.dry_run)
        urls = {
            "google": f"https://www.google.com/search?q={query.replace(' ', '+')}",
            "bing": f"https://www.bing.com/search?q={query.replace(' ', '+')}",
            "duckduckgo": f"https://duckduckgo.com/?q={query.replace(' ', '+')}",
        }
        url = urls.get(engine.lower(), urls["google"])
        return await self.navigate(url)

    # ── Vision-guided interaction ────────────────────────────────────

    async def click_text(self, text: str) -> dict:
        """Find text on page using OCR and click it."""
        log_action("browser", "click_text", text, dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "action": "click_text", "text": text}

        loc = await screen_observer.find_text_location(text)
        if not loc:
            return {"success": False, "error": f"Text '{text}' not found on screen."}

        return await mouse_controller.click(loc[0], loc[1])

    async def click_element(self, description: str) -> dict:
        """Find an element by description using Vision AI and click it."""
        log_action("browser", "click_element", description, dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}

        loc = await screen_observer.find_element_by_description(description)
        if not loc:
            return {"success": False, "error": f"Element '{description}' not found via Vision."}

        return await mouse_controller.click(loc[0], loc[1])

    async def fill_field(self, label: str, value: str) -> dict:
        """Find an input field by label and type a value into it."""
        log_action("browser", "fill_field", f"{label}={value[:20]}", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}

        # Find the label/placeholder text first, then click near it
        loc = await screen_observer.find_text_location(label)
        if not loc:
            return {"success": False, "error": f"Field '{label}' not found on screen."}

        # Click slightly to the right (input is usually after label)
        await mouse_controller.click(loc[0] + 150, loc[1])
        await asyncio.sleep(0.2)
        await keyboard_controller.select_all()
        await keyboard_controller.type_text(value)
        return {"success": True, "action": "fill_field", "label": label}

    async def scroll_page(self, direction: str = "down", amount: int = 3) -> dict:
        """Scroll the browser page."""
        return await mouse_controller.scroll(amount, direction=direction)

    async def get_page_text(self) -> dict:
        """Extract all visible text from the current browser page using OCR."""
        log_action("browser", "get_page_text", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "text": "[DRY RUN: no text]"}
        text, _ = await screen_observer.extract_text()
        return {"success": True, "action": "get_page_text", "text": text}

    async def get_current_url(self) -> str:
        """Attempt to read the current URL from the address bar using OCR."""
        # Focus address bar, select all, copy
        await keyboard_controller.hotkey("ctrl", "l")
        await asyncio.sleep(0.2)
        await keyboard_controller.copy()
        await asyncio.sleep(0.1)
        await keyboard_controller.escape()
        # Read clipboard
        try:
            import pyperclip
            return pyperclip.paste()
        except ImportError:
            return ""


# Global singleton
browser_controller = BrowserController()
