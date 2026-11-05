"""P2 item 2: "open X and type …" types into the window open_app ended up in, never into a pre-existing document.

Every window / keyboard helper is mocked: no keys reach the real desktop.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import automation.ops.keyboard as kb
import brain.pipeline as pipeline
from automation.gateway import ActionRequest, gateway
from brain.intent import Intent
from brain.pipeline import _typing_target, brain
from core.config import settings
from tests.fakes import FakeLLM, calls, install
from tools.base import ToolCallRecord, ToolResult
from tools.registry import registry

DOC = "*YOUR TURN - Kaggle - Notepad"


@pytest.fixture
def on_windows(monkeypatch):
    monkeypatch.setattr(kb, "_IS_WINDOWS", True)
    monkeypatch.setattr(kb, "_FOCUS_SETTLE_SECONDS", 0.0)
    gateway.resume()
    yield kb
    gateway.cancel_session("p2-type")


class Screen:
    """The foreground window as the keyboard op sees it; focus_hwnd moves it (when the window still exists)."""

    def __init__(self, fg, windows, focus_works=True):
        self.fg = fg
        self.windows = dict(windows)
        self.focus_works = focus_works
        self.focused = []

    def foreground_window(self):
        return self.fg, self.windows.get(self.fg, "")

    def focus_hwnd(self, hwnd):
        self.focused.append(hwnd)
        if self.focus_works and hwnd in self.windows:
            self.fg = hwnd
            return True
        return False


async def _type(screen, target, text="hello", origin="fast"):
    with patch.object(kb.impl, "foreground_window", screen.foreground_window), \
            patch.object(kb.impl, "focus_hwnd", screen.focus_hwnd), \
            patch.object(kb.impl, "window_action", return_value=False) as wa, \
            patch("automation.ops.keyboard.send_keys") as sk:
        args = {"text": text, "origin": origin}
        if target is not None:
            args["target"] = target
        res = await gateway.execute(ActionRequest("keyboard.type", args), "p2-type")
    return res, [c.args[0] for c in sk.call_args_list], wa


def _target(kind="new_window", hwnd=42, title="Untitled - Notepad", app="Notepad", first_input=True):
    return {"hwnd": hwnd, "title": title, "kind": kind, "app": app, "first_input": first_input}


# ── keyboard.type with a target window ───────────────────────────────────

async def test_target_in_front_is_typed_into_directly(on_windows):
    screen = Screen(42, {42: "Untitled - Notepad"})
    res, keys, _ = await _type(screen, _target())
    assert res.status == "executed" and keys == ["hello"] and screen.focused == []


async def test_target_behind_another_window_is_focused_first(on_windows):
    screen = Screen(1, {1: "Inbox - Outlook", 42: "Untitled - Notepad"})
    res, keys, _ = await _type(screen, _target())
    assert res.status == "executed" and screen.focused == [42] and keys == ["hello"]


async def test_console_in_front_focuses_the_target_not_alt_tab(on_windows):
    screen = Screen(1, {1: "JARVIS console - Google Chrome", 42: "Untitled - Notepad"})
    res, keys, _ = await _type(screen, _target())
    assert res.status == "executed" and keys == ["hello"]  # no "%{TAB}"


async def test_target_lost_fails_and_sends_no_keys(on_windows):
    screen = Screen(1, {1: "Inbox - Outlook"})  # the Notepad window (42) is gone
    res, keys, _ = await _type(screen, _target())
    assert res.status == "failed" and keys == []
    assert "didn't type" in res.message and "Untitled - Notepad" in res.message


async def test_focus_that_doesnt_take_fails_and_sends_no_keys(on_windows):
    screen = Screen(1, {1: "Inbox - Outlook", 42: "Untitled - Notepad"}, focus_works=False)
    res, keys, _ = await _type(screen, _target())
    assert res.status == "failed" and keys == []


async def test_existing_notepad_with_unsaved_document_gets_a_new_tab_first(on_windows):
    screen = Screen(1, {1: "JARVIS console - Google Chrome", 42: DOC})
    res, keys, _ = await _type(screen, _target(kind="existing", title=DOC), text="groq speed test")
    assert res.status == "executed", res.message
    assert screen.focused == [42] and keys == ["^n", "groq speed test"]


async def test_new_tab_is_only_made_for_the_first_typing_after_the_launch(on_windows):
    screen = Screen(42, {42: DOC})
    _, keys, _ = await _type(screen, _target(kind="existing", title=DOC, first_input=False))
    assert keys == ["hello"]


@pytest.mark.parametrize("kind", ["new_window", "new_tab"])
async def test_notepad_that_already_opened_a_fresh_document_gets_no_extra_tab(on_windows, kind):
    _, keys, _ = await _type(Screen(42, {42: "Untitled - Notepad"}), _target(kind=kind))
    assert keys == ["hello"]


async def test_new_tab_rule_is_only_for_notepad(on_windows):
    screen = Screen(42, {42: "Inbox - Google Chrome"})
    _, keys, _ = await _type(screen, _target(kind="existing", title="Inbox - Google Chrome", app="Chrome"))
    assert keys == ["hello"]


async def test_target_without_a_window_handle_needs_the_app_in_front(on_windows):
    """open_app found no window: typing only goes ahead when the app's window is (or can be brought) in front."""
    target = {"app": "Notepad", "first_input": True}
    res, keys, wa = await _type(Screen(1, {1: "Inbox - Outlook"}), target)
    assert res.status == "failed" and keys == [] and wa.call_args.args == ("Notepad", "focus")
    res, keys, _ = await _type(Screen(42, {42: "Untitled - Notepad"}), target)
    assert res.status == "executed" and keys == ["hello"]


async def test_no_target_keeps_typing_into_the_foreground(on_windows):
    """The owner's own "type X" (no open_app before it) is unchanged."""
    with patch.object(kb.impl, "foreground_title", return_value="Inbox - Outlook"), \
            patch("automation.ops.keyboard.send_keys") as sk:
        res = await gateway.execute(ActionRequest("keyboard.type", {"text": "hi", "origin": "fast"}), "p2-type")
    assert res.status == "executed" and [c.args[0] for c in sk.call_args_list] == ["hi"]


# ── the agent passes open_app's window to the following type_text ────────

def _opened(window=None, success=True, ready=True):
    data = {"app": "Notepad", "ready": ready}
    if window:
        data["window"] = window
    step = {"op": "app.open", "args": {"name": "notepad"}, "success": success, "message": "Opened Notepad.", "data": data}
    return ToolCallRecord("open_app", {"name": "notepad"},
                          ToolResult(success, "Opened Notepad.", {"steps": [step], "risk": "low"}))


WIN = {"hwnd": 42, "title": DOC, "kind": "existing"}


def test_typing_target_is_the_window_open_app_reported():
    assert _typing_target([_opened(WIN)]) == {**WIN, "app": "Notepad", "first_input": True}


def test_typing_target_after_an_earlier_typing_is_not_first_input():
    typed = ToolCallRecord("type_text", {"text": "a"}, ToolResult.ok("Typed it."))
    assert _typing_target([_opened(WIN), typed])["first_input"] is False


def test_focus_window_after_open_app_drops_the_target():
    focused = ToolCallRecord("focus_window", {"name": "chrome"}, ToolResult.ok("Switched to 'chrome'."))
    assert _typing_target([_opened(WIN), focused]) is None


def test_no_open_app_this_turn_means_no_target():
    assert _typing_target([]) is None


def test_open_app_without_a_window_still_names_the_app():
    assert _typing_target([_opened(ready=False)]) == {"app": "Notepad", "first_input": True}
    failed = ToolCallRecord("open_app", {"name": "notepad"}, ToolResult.error("Application 'notepad' was not found."))
    assert _typing_target([failed]) == {"app": "notepad", "first_input": True}


async def test_open_notepad_and_type_goes_to_a_new_tab_not_the_unsaved_document(monkeypatch, on_windows):
    """The live failure: Notepad already open with an unsaved document; the text must land in a new tab."""
    monkeypatch.setattr(settings, "CONFIRM_MEDIUM_RISK", False)
    monkeypatch.setattr(pipeline, "analyze", AsyncMock(return_value=Intent(
        request="Open Notepad and type 'groq speed test'", kind="action", continues_task=False, single_step=False,
        tool_groups=["keyboard"], needs_screen=False, source="llm")))
    monkeypatch.setattr(registry.get("open_app"), "run", AsyncMock(return_value=_opened(WIN).result))
    install(monkeypatch, FakeLLM([calls(("open_app", {"name": "notepad"}), ("type_text", {"text": "groq speed test"}))]))
    screen = Screen(1, {1: "JARVIS console - Google Chrome", 42: DOC})
    with patch.object(kb.impl, "foreground_window", screen.foreground_window), \
            patch.object(kb.impl, "focus_hwnd", screen.focus_hwnd), \
            patch("automation.ops.keyboard.send_keys") as sk:
        res = await brain.handle("open notepad and type groq speed test", "p2-type")
    assert res.status == "ok", res.response
    assert screen.focused == [42] and [c.args[0] for c in sk.call_args_list] == ["^n", "groq speed test"]


async def test_agent_typing_with_lost_target_fails_without_keys(monkeypatch, on_windows):
    monkeypatch.setattr(settings, "CONFIRM_MEDIUM_RISK", False)
    monkeypatch.setattr(pipeline, "analyze", AsyncMock(return_value=Intent(
        request="Open Notepad and type 'hi'", kind="action", continues_task=False, single_step=False,
        tool_groups=["keyboard"], needs_screen=False, source="llm")))
    monkeypatch.setattr(registry.get("open_app"), "run", AsyncMock(return_value=_opened(
        {"hwnd": 42, "title": "Untitled - Notepad", "kind": "new_window"}).result))
    install(monkeypatch, FakeLLM([calls(("open_app", {"name": "notepad"}), ("type_text", {"text": "hi"}))]))
    screen = Screen(1, {1: "Inbox - Outlook"})  # Notepad closed before typing
    with patch.object(kb.impl, "foreground_window", screen.foreground_window), \
            patch.object(kb.impl, "focus_hwnd", screen.focus_hwnd), \
            patch("automation.ops.keyboard.send_keys") as sk:
        res = await brain.handle("open notepad and type hi", "p2-type")
    sk.assert_not_called()
    assert "didn't type" in str(res.tools) or "didn't type" in res.response
