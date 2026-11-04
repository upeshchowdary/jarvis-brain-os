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
        res = await gateway.execute(ActionRequest("keyboard.press", {"keys": keys, "origin": "fast"}), "kb-test")
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
        res = await gateway.execute(ActionRequest("keyboard.type", {"text": "a+b", "origin": "fast"}), "kb-test")
    assert res.status == "executed", res
    sk.assert_called_once_with("a{+}b", with_spaces=True)


async def test_type_escapes_all_pywinauto_specials():
    with patch("automation.ops.keyboard.send_keys") as sk:
        await gateway.execute(ActionRequest("keyboard.type", {"text": "^%~(){}[]", "origin": "fast"}), "kb-test")
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


# ── fix round 1 ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("utterance,text", [
    ("type Hello.", "Hello."),
    ("type see you tomorrow thanks", "see you tomorrow thanks"),
    ("type please call me", "please call me"),
    ("type Hello, World!", "Hello, World!"),
    ("Jarvis, type Mixed CASE text", "Mixed CASE text"),
    ("please type Thanks please", "Thanks please"),
])
def test_type_text_is_raw(utterance, text):
    d = fast_route(utterance)
    assert d.tool == "type_text" and d.args == {"text": text}


@pytest.mark.parametrize("utterance", [
    "type of file is this?", "type of file is this", "what type is this?", "type 2 diabetes symptoms",
    "type 1 diabetes", "type something?", "type",
])
def test_type_is_not_a_catch_all(utterance):
    d = fast_route(utterance)
    assert d is None or d.tool != "type_text"


def test_last_tab_vs_previous_tab():
    assert fast_route("last tab").args == {"keys": "ctrl+9"}
    assert fast_route("previous tab").args == {"keys": "ctrl+shift+tab"}


@pytest.mark.parametrize("utterance,tool", [
    ("close chrome", "close_app"),
    ("open chrome", "open_app"),
    ("go to github.com", "open_website"),
    ("next track", "media_control"),
    ("stop", "automation_control"),
])
def test_existing_rules_still_win(utterance, tool):
    assert fast_route(utterance).tool == tool


def test_new_tool_flags():
    from tools.registry import registry

    for name in ("press_keys", "type_text", "focus_window"):
        t = registry.get(name)
        assert t.side_effect is True and t.final is True


# ── I5: the user's own words (fast path) are LOW; the LLM agent's key presses / typing are MEDIUM ──

@pytest.fixture
def on_windows(monkeypatch):
    import automation.ops.keyboard as kb

    monkeypatch.setattr(kb, "_IS_WINDOWS", True)
    gateway.resume()
    yield kb
    gateway.cancel_session("kb-risk")


def test_keyboard_risk_depends_on_origin():
    from automation.gateway import Risk

    for op in ("keyboard.press", "keyboard.type"):
        assert gateway.classify(ActionRequest(op, {"origin": "fast"})) == Risk.LOW
        assert gateway.classify(ActionRequest(op, {"origin": "llm"})) == Risk.MEDIUM
        assert gateway.classify(ActionRequest(op, {})) == Risk.MEDIUM  # unknown caller: the safe default
    assert gateway.classify(ActionRequest("window.focus", {"name": "x"})) == Risk.LOW


async def test_fast_path_typing_runs_without_confirmation_on_defaults(on_windows):
    from brain.pipeline import brain

    with patch("automation.ops.keyboard.send_keys") as sk:
        res = await brain.handle("type hello there", "kb-risk")
    assert res.route == "fast_path" and res.status == "ok", res.response
    sk.assert_called_once_with("hello there", with_spaces=True)


@pytest.mark.parametrize("confirm_medium, status", [(True, "needs_confirmation"), (False, "ok")])
async def test_llm_typing_needs_confirmation_when_medium_confirms(monkeypatch, on_windows, confirm_medium, status):
    from unittest.mock import AsyncMock

    import brain.pipeline as pipeline
    from brain.intent import Intent
    from core.config import settings
    from tests.fakes import FakeLLM, call, install

    monkeypatch.setattr(settings, "CONFIRM_MEDIUM_RISK", confirm_medium)
    monkeypatch.setattr(pipeline, "analyze", AsyncMock(return_value=Intent(
        request="Put the words hello there into the focused box", kind="action", continues_task=False, single_step=True,
        tool_groups=["keyboard"], needs_screen=False, source="llm")))
    install(monkeypatch, FakeLLM([call("type_text", text="hello there")]))
    with patch("automation.ops.keyboard.send_keys") as sk:
        res = await brain_handle("could you put hello there in the box", "kb-risk")
    assert res.status == status, res.response
    assert sk.called is (not confirm_medium)


async def brain_handle(query, sid):
    from brain.pipeline import brain

    return await brain.handle(query, sid)


async def test_rewritten_typing_is_not_the_users_own_words(monkeypatch, on_windows):
    """The intent step's rewrite ("do it again" → "type …") may reach the fast path, but it is MEDIUM, not LOW."""
    from unittest.mock import AsyncMock

    import brain.pipeline as pipeline
    from brain.intent import Intent

    monkeypatch.setattr(pipeline, "analyze", AsyncMock(return_value=Intent(
        request="type hello there", kind="action", continues_task=True, single_step=True,
        tool_groups=["keyboard"], needs_screen=False, source="llm")))
    with patch("automation.ops.keyboard.send_keys") as sk:
        res = await brain_handle("do that one more time", "kb-risk")
    assert res.route == "fast_path" and res.status == "needs_confirmation", res.response
    sk.assert_not_called()


# ── I8: from the JARVIS console, keys go to the window the user was in, not the console itself ──

async def _press_from(titles, keys="ctrl+w"):
    with patch("automation.ops.keyboard.impl.foreground_title", side_effect=list(titles)) as fg, \
            patch("automation.ops.keyboard.send_keys") as sk:
        res = await gateway.execute(ActionRequest("keyboard.press", {"keys": keys, "origin": "fast"}), "kb-console")
    return res, sk, fg


async def test_console_in_front_switches_to_previous_window_first(on_windows):
    res, sk, _ = await _press_from(["JARVIS console - Google Chrome", "Inbox - Outlook"])
    assert res.status == "executed", res
    assert [c.args[0] for c in sk.call_args_list] == ["%{TAB}", "^w"]


async def test_console_with_no_other_window_fails_clearly(on_windows):
    res, sk, _ = await _press_from(["JARVIS console - Google Chrome", "JARVIS console - Google Chrome"])
    assert res.status == "failed" and "Click the window you want first." in res.message
    assert [c.args[0] for c in sk.call_args_list] == ["%{TAB}"]  # the keys never reached the console


async def test_other_window_in_front_gets_keys_directly(on_windows):
    res, sk, fg = await _press_from(["Untitled - Notepad"], keys="f5")
    assert res.status == "executed"
    assert [c.args[0] for c in sk.call_args_list] == ["{F5}"] and fg.call_count == 1


async def test_typing_from_console_goes_to_previous_window(on_windows):
    with patch("automation.ops.keyboard.impl.foreground_title", side_effect=["JARVIS console", "Untitled - Notepad"]), \
            patch("automation.ops.keyboard.send_keys") as sk:
        res = await gateway.execute(ActionRequest("keyboard.type", {"text": "hi", "origin": "fast"}), "kb-console")
    assert res.status == "executed"
    assert [c.args[0] for c in sk.call_args_list] == ["%{TAB}", "hi"]


def test_console_title_matches_the_page():
    from pathlib import Path

    import automation.ops.keyboard as kb

    html = (Path(__file__).resolve().parents[1] / "app" / "static" / "web" / "console.html").read_text(encoding="utf-8")
    assert f"<title>{kb.CONSOLE_TITLE}</title>" in html


# ── P1 item 4: "type it again" is a reference to earlier text, not dictation ──

@pytest.mark.parametrize("utterance", [
    "type it again", "type that", "type it", "Type it again.", "jarvis type that again", "please type it again",
    "type the same", "type the same thing", "type same thing", "type again", "type that one more time",
])
def test_type_reference_is_not_fast_pathed(utterance):
    d = fast_route(utterance)
    assert d is None or d.tool != "type_text", d


@pytest.mark.parametrize("utterance,text", [
    ("type it is raining", "it is raining"),
    ("type that's all folks", "that's all folks"),
    ("type again and again", "again and again"),
])
def test_type_text_starting_with_a_reference_word_is_still_typed(utterance, text):
    d = fast_route(utterance)
    assert d is not None and d.tool == "type_text" and d.args == {"text": text}


async def test_type_it_again_goes_to_the_intent_step(monkeypatch, on_windows):
    """The intent step resolves "it" from history; a quoted rewrite types the text without the quotes."""
    from unittest.mock import AsyncMock

    import brain.pipeline as pipeline
    from brain.intent import Intent

    analyze = AsyncMock(return_value=Intent(
        request="Type 'hello from jarvis'", kind="action", continues_task=True, single_step=True,
        tool_groups=["keyboard"], needs_screen=False, source="llm"))
    monkeypatch.setattr(pipeline, "analyze", analyze)
    try:
        with patch("automation.ops.keyboard.send_keys") as sk:
            res = await brain_handle("type it again", "kb-type-again")
    finally:
        gateway.cancel_session("kb-type-again")
    analyze.assert_awaited_once()
    assert res.route == "fast_path" and res.status == "needs_confirmation", res.response
    assert res.tools[0]["arguments"] == {"text": "hello from jarvis"}
    sk.assert_not_called()


# ── P1 fix round 1: reference phrasings with polite words / wake word / commas ──

@pytest.mark.parametrize("utterance", [
    "type it again please", "type it again, jarvis", "type that once again", "type that one", "type this one",
    "type it again please jarvis", "type that, please", "Type it once again!", "type the same thing again, thanks",
])
def test_type_reference_with_polite_words_is_not_fast_pathed(utterance):
    d = fast_route(utterance)
    assert d is None or d.tool != "type_text", d


@pytest.mark.parametrize("utterance,text", [
    ("type it is raining", "it is raining"),
    ("type that's all folks", "that's all folks"),
    ("type again and again", "again and again"),
    ("type that one is mine", "that one is mine"),
    ("type please call me", "please call me"),
])
def test_type_dictation_still_literal_after_reference_normalisation(utterance, text):
    d = fast_route(utterance)
    assert d is not None and d.tool == "type_text" and d.args == {"text": text}
