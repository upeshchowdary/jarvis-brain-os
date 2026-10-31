"""Instant keyboard / window-focus operations (no LLM, ~10-50 ms).

All of them act on the *foreground* window via ``pywinauto.keyboard.send_keys``.
The caller passes a normalised key string such as ``"ctrl+shift+t"``; it is mapped to
pywinauto syntax here. Unknown keys are rejected: arbitrary send_keys strings never get through.
"""

from __future__ import annotations

import asyncio
import re
import sys
from typing import Any, Dict, Optional

from automation.gateway import OpResult, Preview, Risk, gateway
from automation.ops import system_impl as impl

_IS_WINDOWS = sys.platform == "win32"

_MODIFIERS = {"ctrl": "^", "control": "^", "shift": "+", "alt": "%"}
_NAMED = {
    "enter": "{ENTER}", "return": "{ENTER}", "escape": "{ESC}", "esc": "{ESC}", "tab": "{TAB}", "space": "{SPACE}",
    "backspace": "{BACKSPACE}", "delete": "{DEL}", "del": "{DEL}", "up": "{UP}", "down": "{DOWN}", "left": "{LEFT}",
    "right": "{RIGHT}", "home": "{HOME}", "end": "{END}", "pageup": "{PGUP}", "pagedown": "{PGDN}",
    **{f"f{i}": f"{{F{i}}}" for i in range(1, 13)},
}
_SPECIAL = re.compile(r"([{}()+^%~\[\]])")


def to_send_keys(keys: str) -> Optional[str]:
    """``"ctrl+shift+t"`` -> ``"^+t"``. None when any part is not a known key."""
    parts = [p.strip().lower() for p in str(keys or "").split("+")]
    if not parts or any(not p for p in parts):
        return None
    *mods, key = parts
    prefix = ""
    for m in mods:
        if m not in _MODIFIERS:
            return None
        prefix += _MODIFIERS[m]
    if key in _NAMED:
        return prefix + _NAMED[key]
    if len(key) == 1 and key.isascii() and key.isalnum():
        return prefix + key
    return None


def escape_text(text: str) -> str:
    return _SPECIAL.sub(r"{\1}", text)


async def _run(func, *args, **kwargs) -> Any:
    return await asyncio.to_thread(lambda: func(*args, **kwargs))


try:  # module-level so tests can patch it; unavailable off Windows
    from pywinauto.keyboard import send_keys  # type: ignore
except Exception:  # noqa: BLE001
    send_keys = None  # type: ignore[assignment]


def _send(seq: str, **kwargs: Any) -> None:
    if send_keys is None:
        raise RuntimeError("pywinauto is not available")
    send_keys(seq, **kwargs)


def _press_preview(args: Dict[str, Any]) -> Preview:
    keys = str(args.get("keys", ""))
    if to_send_keys(keys) is None:
        return Preview(f"Unknown key combination '{keys}'.", ok=False)
    return Preview(f"Press {keys} in the active window.", ok=True)


@gateway.operation("keyboard.press", Risk.LOW, "Press a key or key combination in the active window", preview=_press_preview)
async def press(keys: str = "") -> OpResult:
    if not _IS_WINDOWS:
        return OpResult.fail("Keyboard control is only available on Windows.")
    seq = to_send_keys(keys)
    if seq is None:
        return OpResult.fail(f"I don't know the key '{keys}'.")
    try:
        await _run(_send, seq)
    except Exception as e:  # noqa: BLE001
        return OpResult.fail(f"Couldn't press {keys}: {e}")
    return OpResult.ok(f"Pressed {keys}.")


def _type_preview(args: Dict[str, Any]) -> Preview:
    text = str(args.get("text", ""))
    shown = text if len(text) <= 60 else text[:57] + "..."
    return Preview(f'Type "{shown}" into the active window.', ok=bool(text))


@gateway.operation("keyboard.type", Risk.LOW, "Type text into the active window", preview=_type_preview)
async def type_text(text: str = "") -> OpResult:
    if not _IS_WINDOWS:
        return OpResult.fail("Keyboard control is only available on Windows.")
    if not text:
        return OpResult.fail("Nothing to type.")
    try:
        await _run(_send, escape_text(str(text)), with_spaces=True)
    except Exception as e:  # noqa: BLE001
        return OpResult.fail(f"Couldn't type that: {e}")
    return OpResult.ok("Typed it.")


def _focus_preview(args: Dict[str, Any]) -> Preview:
    name = str(args.get("name", "")).strip()
    return Preview(f"Switch to the window matching '{name}'.", ok=bool(name))


@gateway.operation("window.focus", Risk.LOW, "Bring a window to the front by title/app name", preview=_focus_preview)
async def focus(name: str = "") -> OpResult:
    if not _IS_WINDOWS:
        return OpResult.fail("Window control is only available on Windows.")
    name = str(name or "").strip()
    if not name:
        return OpResult.fail("Which window?")
    ok = await _run(impl.window_action, name, "focus")
    return OpResult(ok, f"Switched to '{name}'." if ok else f"Couldn't find a window matching '{name}'.")
