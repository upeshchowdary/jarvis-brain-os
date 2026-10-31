"""Keyboard / window fast path: router rules, key mapping and the gateway ops (send_keys mocked)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from automation.gateway import ActionRequest, gateway
from brain.router import CONTEXT_WORDS, fast_route


@pytest.mark.parametrize("utterance,tool,args", [
    ("scroll down", "press_keys", {"keys": "pagedown"}),
    ("scroll up", "press_keys", {"keys": "pageup"}),
    ("page down", "press_keys", {"keys": "pagedown"}),
    ("page up", "press_keys", {"keys": "pageup"}),
    ("go to the top", "press_keys", {"keys": "ctrl+home"}),
    ("go to bottom", "press_keys", {"keys": "ctrl+end"}),
    ("next tab", "press_keys", {"keys": "ctrl+tab"}),
    ("previous tab", "press_keys", {"keys": "ctrl+shift+tab"}),
    ("go to tab 3", "press_keys", {"keys": "ctrl+3"}),
    ("switch to tab 2", "press_keys", {"keys": "ctrl+2"}),
    ("go to tab last", "press_keys", {"keys": "ctrl+9"}),
    ("new tab", "press_keys", {"keys": "ctrl+t"}),
    ("close tab", "press_keys", {"keys": "ctrl+w"}),
    ("close this tab", "press_keys", {"keys": "ctrl+w"}),
    ("reopen closed tab", "press_keys", {"keys": "ctrl+shift+t"}),
    ("reopen tab", "press_keys", {"keys": "ctrl+shift+t"}),
    ("go back", "press_keys", {"keys": "alt+left"}),
    ("go forward", "press_keys", {"keys": "alt+right"}),
    ("refresh", "press_keys", {"keys": "f5"}),
    ("reload", "press_keys", {"keys": "f5"}),
    ("select all", "press_keys", {"keys": "ctrl+a"}),
    ("copy", "press_keys", {"keys": "ctrl+c"}),
    ("paste", "press_keys", {"keys": "ctrl+v"}),
    ("undo typing", "press_keys", {"keys": "ctrl+z"}),
    ("redo", "press_keys", {"keys": "ctrl+y"}),
    ("press enter", "press_keys", {"keys": "enter"}),
    ("press escape", "press_keys", {"keys": "escape"}),
    ("press the space key", "press_keys", {"keys": "space"}),
    ("press f5", "press_keys", {"keys": "f5"}),
    ("press down", "press_keys", {"keys": "down"}),
    ("please scroll down", "press_keys", {"keys": "pagedown"}),
    ("minimize this", "window_control", {"action": "minimize"}),
    ("maximize this", "window_control", {"action": "maximize"}),
    ("switch to chrome", "focus_window", {"name": "chrome"}),
    ("switch to notepad", "focus_window", {"name": "notepad"}),
])
def test_utterance_routes(utterance, tool, args):
    d = fast_route(utterance)
    assert d is not None, utterance
    assert d.tool == tool and d.args == args


def test_type_preserves_case():
    d = fast_route("type Hello World")
    assert d.tool == "type_text" and d.args == {"text": "Hello World"}


@pytest.mark.parametrize("model", ["groq", "gemini", "claude", "ollama", "local"])
def test_switch_to_model_still_switches_model(model):
    d = fast_route(f"switch to {model}")
    assert d.tool == "switch_model" and d.args == {"model": model}


@pytest.mark.parametrize("word", sorted(CONTEXT_WORDS))
def test_bare_context_words_are_not_keyboard_rules(word):
    d = fast_route(word, task_active=True)
    assert d is None
    d = fast_route(word)
    assert d is None or d.tool not in ("press_keys", "type_text", "focus_window")


def test_unknown_press_key_not_routed():
    assert fast_route("press banana") is None


def test_bare_undo_stays_action_undo():
    assert fast_route("undo").tool == "undo_last_action"


def test_switch_to_unknown_target_still_goes_to_model_switch():
    assert fast_route("switch to zzzfoo").tool == "switch_model"


# ── gateway ops (send_keys mocked) ───────────────────────────────────────

async def _press(keys):
    with patch("automation.ops.keyboard.send_keys") as sk:
        res = await gateway.execute(ActionRequest("keyboard.press", {"keys": keys}), "kb-test")
    return res, sk


@pytest.mark.parametrize("keys,seq", [
    ("ctrl+tab", "^{TAB}"),
    ("ctrl+shift+t", "^+t"),
    ("pagedown", "{PGDN}"),
    ("alt+left", "%{LEFT}"),
    ("f5", "{F5}"),
    ("ctrl+1", "^1"),
    ("ctrl+shift+tab", "^+{TAB}"),
])
async def test_press_calls_send_keys(keys, seq):
    res, sk = await _press(keys)
    assert res.status == "executed", res
    sk.assert_called_once_with(seq)


@pytest.mark.parametrize("keys", ["ctrl+banana", "{ENTER}{ENTER}", "", "ctrl+", "win+r"])
async def test_press_rejects_unknown_keys(keys):
    res, sk = await _press(keys)
    assert res.status != "executed"
    sk.assert_not_called()


async def test_type_escapes_special_chars():
    with patch("automation.ops.keyboard.send_keys") as sk:
        res = await gateway.execute(ActionRequest("keyboard.type", {"text": "a+b"}), "kb-test")
    assert res.status == "executed", res
    sk.assert_called_once_with("a{+}b", with_spaces=True)


async def test_type_escapes_all_pywinauto_specials():
    with patch("automation.ops.keyboard.send_keys") as sk:
        await gateway.execute(ActionRequest("keyboard.type", {"text": "^%~(){}[]"}), "kb-test")
    sk.assert_called_once_with("{^}{%}{~}{(}{)}{{}{}}{[}{]}", with_spaces=True)


async def test_window_focus_uses_window_action():
    with patch("automation.ops.keyboard.impl.window_action", return_value=True) as wa:
        res = await gateway.execute(ActionRequest("window.focus", {"name": "chrome"}), "kb-test")
    assert res.status == "executed", res
    wa.assert_called_once_with("chrome", "focus")


async def test_dry_run_previews_describe_keys(monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "DRY_RUN", True)
    with patch("automation.ops.keyboard.send_keys") as sk:
        res = await gateway.execute(ActionRequest("keyboard.press", {"keys": "ctrl+tab"}), "kb-test")
        res2 = await gateway.execute(ActionRequest("keyboard.type", {"text": "hi"}), "kb-test")
    sk.assert_not_called()
    assert "ctrl+tab" in str(res.message) + str(res.preview or "")
    assert "hi" in str(res2.message) + str(res2.preview or "")


def test_keyboard_group_selects_tools():
    from tools.registry import registry
    from tools.selection import select_tools

    names = {t.name for t in select_tools("scroll down", registry.all())}
    assert {"press_keys", "type_text", "focus_window"} <= names
