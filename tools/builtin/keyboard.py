"""Instant keyboard / window-focus tools (act on the foreground window). Always through the ActionGateway.

Key presses and typing pass ``ctx.origin`` on: LOW from the fast path (the user's own words), MEDIUM from the agent.
"""

from __future__ import annotations

from typing import Any

from automation.gateway import ActionRequest, Risk
from tools.base import Tool, ToolContext, ToolResult, obj, via_gateway
from tools.registry import register


class _KeyboardTool(Tool):
    side_effect = True
    risk = Risk.LOW
    final = True


@register
class PressKeys(_KeyboardTool):
    name = "press_keys"
    description = (
        "Press a key or shortcut in the active window. keys is lower-case and '+'-joined, e.g. 'ctrl+tab', "
        "'ctrl+shift+t', 'pagedown', 'alt+left', 'f5', 'enter', 'escape', 'ctrl+1'."
    )
    parameters = obj({"keys": {"type": "string"}}, ["keys"])

    async def run(self, ctx: ToolContext, keys: str = "", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("keyboard.press", {"keys": keys, "origin": ctx.origin}))


@register
class TypeText(_KeyboardTool):
    name = "type_text"
    description = "Type the given text into the active window exactly as written."
    parameters = obj({"text": {"type": "string"}}, ["text"])

    async def run(self, ctx: ToolContext, text: str = "", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("keyboard.type", {"text": text, "origin": ctx.origin}))


@register
class FocusWindow(_KeyboardTool):
    name = "focus_window"
    description = "Bring an open window/app to the front by (part of) its title or app name, e.g. 'chrome', 'notepad'."
    parameters = obj({"name": {"type": "string"}}, ["name"])

    async def run(self, ctx: ToolContext, name: str = "", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("window.focus", {"name": name}))
