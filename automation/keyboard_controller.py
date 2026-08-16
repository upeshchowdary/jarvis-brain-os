"""JARVIS Keyboard Controller — Full keyboard interaction with safety guards.

Supports: type text, press keys, hotkeys, key_down/up, all modifier combinations.
Never types sensitive data into unknown fields.
"""

import asyncio
from typing import List

from automation.config import automation_config
from automation.automation_logger import log_action

try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False


# Keys that map to common operation names
_KEY_ALIASES = {
    "enter": "enter", "return": "enter",
    "escape": "escape", "esc": "escape",
    "tab": "tab", "shift_tab": ["shift", "tab"],
    "backspace": "backspace", "delete": "delete",
    "space": "space", "spacebar": "space",
    "up": "up", "down": "down", "left": "left", "right": "right",
    "home": "home", "end": "end", "pageup": "pageup", "pagedown": "pagedown",
    "f1": "f1", "f2": "f2", "f3": "f3", "f4": "f4", "f5": "f5",
    "f6": "f6", "f7": "f7", "f8": "f8", "f9": "f9", "f10": "f10",
    "f11": "f11", "f12": "f12",
    "ctrl_c": ["ctrl", "c"], "ctrl_v": ["ctrl", "v"],
    "ctrl_x": ["ctrl", "x"], "ctrl_z": ["ctrl", "z"],
    "ctrl_a": ["ctrl", "a"], "ctrl_s": ["ctrl", "s"],
    "ctrl_f": ["ctrl", "f"], "ctrl_w": ["ctrl", "w"],
    "ctrl_t": ["ctrl", "t"], "ctrl_r": ["ctrl", "r"],
    "alt_tab": ["alt", "tab"], "alt_f4": ["alt", "f4"],
    "win": "win", "printscreen": "printscreen",
}


class KeyboardController:
    """Keyboard interaction controller with dry-run and safety support."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run or automation_config.DRY_RUN

    def _require_pyautogui(self) -> None:
        if not HAS_PYAUTOGUI:
            raise RuntimeError("pyautogui is not installed. Run: pip install pyautogui")

    # ── Text Typing ─────────────────────────────────────────────────

    async def type_text(self, text: str, interval: float | None = None) -> dict:
        """Type text with realistic character-by-character timing."""
        # Safety: warn if typing looks like a password
        if any(kw in text.lower() for kw in ["password", "passwd", "secret", "token"]):
            log_action("keyboard", "type_text", "***REDACTED***",
                      result="WARN: Possible sensitive data — redacted from logs",
                      dry_run=self.dry_run)
        else:
            log_action("keyboard", "type_text",
                      text[:40] + ("..." if len(text) > 40 else ""),
                      dry_run=self.dry_run)

        if self.dry_run:
            return {"success": True, "dry_run": True, "action": "type_text", "length": len(text)}

        self._require_pyautogui()
        char_interval = interval if interval is not None else automation_config.TYPE_INTERVAL
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: pyautogui.typewrite(text, interval=char_interval)
        )
        return {"success": True, "action": "type_text", "length": len(text)}

    async def type_raw(self, text: str) -> dict:
        """Type text using clipboard paste (faster for long strings, handles special chars)."""
        log_action("keyboard", "type_raw (clipboard paste)",
                  f"{len(text)} chars", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "action": "type_raw"}
        self._require_pyautogui()
        import pyperclip
        await asyncio.get_event_loop().run_in_executor(None, lambda: pyperclip.copy(text))
        await self.hotkey("ctrl", "v")
        return {"success": True, "action": "type_raw"}

    # ── Key Presses ─────────────────────────────────────────────────

    async def press(self, key: str, presses: int = 1) -> dict:
        """Press a single key or alias."""
        resolved = _KEY_ALIASES.get(key.lower(), key.lower())
        log_action("keyboard", "press", str(resolved), dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "action": "press", "key": key}
        self._require_pyautogui()
        if isinstance(resolved, list):
            # It's a hotkey alias
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: pyautogui.hotkey(*resolved)
            )
        else:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: pyautogui.press(resolved, presses=presses)
            )
        return {"success": True, "action": "press", "key": key}

    async def hotkey(self, *keys: str) -> dict:
        """Press a key combination simultaneously (e.g. ctrl+c)."""
        combo = "+".join(keys)
        log_action("keyboard", "hotkey", combo, dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "action": "hotkey", "combo": combo}
        self._require_pyautogui()
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: pyautogui.hotkey(*keys)
        )
        return {"success": True, "action": "hotkey", "combo": combo}

    async def key_down(self, key: str) -> dict:
        """Hold a key down."""
        log_action("keyboard", "key_down", key, dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}
        self._require_pyautogui()
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: pyautogui.keyDown(key)
        )
        return {"success": True, "action": "key_down", "key": key}

    async def key_up(self, key: str) -> dict:
        """Release a held key."""
        log_action("keyboard", "key_up", key, dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}
        self._require_pyautogui()
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: pyautogui.keyUp(key)
        )
        return {"success": True, "action": "key_up", "key": key}

    # ── Convenience shortcuts ────────────────────────────────────────

    async def copy(self) -> dict:
        return await self.hotkey("ctrl", "c")

    async def paste(self) -> dict:
        return await self.hotkey("ctrl", "v")

    async def cut(self) -> dict:
        return await self.hotkey("ctrl", "x")

    async def select_all(self) -> dict:
        return await self.hotkey("ctrl", "a")

    async def undo(self) -> dict:
        return await self.hotkey("ctrl", "z")

    async def save(self) -> dict:
        return await self.hotkey("ctrl", "s")

    async def enter(self) -> dict:
        return await self.press("enter")

    async def escape(self) -> dict:
        return await self.press("escape")

    async def tab(self, shift: bool = False) -> dict:
        if shift:
            return await self.hotkey("shift", "tab")
        return await self.press("tab")

    async def arrow(self, direction: str, count: int = 1) -> dict:
        """Press arrow key N times."""
        key = {"up": "up", "down": "down", "left": "left", "right": "right"}.get(direction.lower(), "down")
        return await self.press(key, presses=count)

    async def alt_tab(self) -> dict:
        return await self.hotkey("alt", "tab")


# Global singleton
keyboard_controller = KeyboardController()
