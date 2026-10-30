"""Positive routing tests: real commands must reach the right tool with the right arguments.

Fast-path commands are checked exactly (tool + key arguments). Commands that need the
LLM agent (multi-step UI work) must at least be recognised as imperative so that the
agent is offered action tools.
"""

import pytest

from automation.engine.messaging_handler import _detect_app_from_text, _heuristic_contact
from automation.engine.planner import _detect_messaging_app, task_planner
from brain import router
from tools.builtin.meta import resolve_alias


# ── Communication understanding (contact/app extraction used by the messaging fallback) ──
@pytest.mark.parametrize("query,expected_app,expected_contact", [
    ("send a mail to clash938134@gmail.com invite him to my party write this mail in formal way", "gmail", "clash938134@gmail.com"),
    ("email john.doe@company.com about the project status update", "gmail", "john.doe@company.com"),
    ("open gmail and compose mail to professor@university.edu asking for extension", "gmail", "professor@university.edu"),
    ("write an email to support@github.com regarding account recovery", "gmail", "support@github.com"),
    ("send mail to alex@startup.io with pitch deck details", "gmail", "alex@startup.io"),
    ("send a formal mail to hr@company.com applying for python developer role", "gmail", "hr@company.com"),
    ("mail test.user@gmail.com that the meeting is rescheduled to 4 PM", "gmail", "test.user@gmail.com"),
    ("open whatsapp and msg karthikeya hi", "whatsapp", "Karthikeya"),
    ("send whatsapp message to Mom asking what is for dinner", "whatsapp", "Mom"),
    ("tell Rahul on whatsapp that I will be late by 15 minutes", "whatsapp", "Rahul"),
    ("open telegram and message Rohit check if college is open tomorrow", "telegram", "Rohit"),
    ("send telegram message to Priya sending project documentation", "telegram", "Priya"),
    ("dm Alex on discord join general voice channel", "discord", "Alex"),
    ("message Team on slack please review pull request 42", "slack", "Team"),
    ("tell Vikram on telegram to call me urgently", "telegram", "Vikram"),
])
def test_communication_understanding(query, expected_app, expected_contact):
    assert router.is_imperative(query), "messaging requests must be offered action tools"
    assert _detect_app_from_text(query) == expected_app
    assert _detect_messaging_app(query) == expected_app
    assert expected_contact.lower() in _heuristic_contact(query).lower()


# ── Exact fast-path commands ─────────────────────────────────────────────────
@pytest.mark.parametrize("query,tool,args", [
    ("set brightness to 80", "set_brightness", {"action": "set", "level": 80}),
    ("brightness to 50", "set_brightness", {"action": "set", "level": 50}),
    ("increase brightness", "set_brightness", {"action": "up"}),
    ("dim the screen", "set_brightness", {"action": "down"}),
    ("turn up the volume", "set_volume", {"action": "up"}),
    ("volume down", "set_volume", {"action": "down"}),
    ("set volume to 35%", "set_volume", {"action": "set", "level": 35}),
    ("increase volume by 20 percent", "set_volume", {"action": "up", "step": 20}),
    ("mute sound", "set_volume", {"action": "mute"}),
    ("unmute audio", "set_volume", {"action": "mute"}),
    ("show desktop", "window_control", {"action": "show_desktop"}),
    ("minimize window", "window_control", {"action": "minimize"}),
    ("maximize this window", "window_control", {"action": "maximize"}),
    ("close active window", "window_control", {"action": "close"}),
    ("restore window", "window_control", {"action": "restore"}),
    ("snap window left", "window_control", {"action": "snap", "direction": "left"}),
    ("snap window right", "window_control", {"action": "snap", "direction": "right"}),
    ("switch window", "window_control", {"action": "switch"}),
    ("list open windows", "list_windows", {}),
    ("lock computer", "lock_screen", {}),
    ("lock the screen please", "lock_screen", {}),
    ("put laptop to sleep", "power_control", {"action": "sleep"}),
    ("restart my pc", "power_control", {"action": "restart"}),
    ("shut down the computer", "power_control", {"action": "shutdown"}),
    ("cancel the shutdown", "cancel_shutdown", {}),
    ("take a screenshot", "take_screenshot", {}),
    ("turn off wifi", "network_toggle", {"device": "wifi", "enable": False}),
    ("turn bluetooth on", "network_toggle", {"device": "bluetooth", "enable": True}),
    ("empty the recycle bin", "empty_recycle_bin", {}),
    ("open Chrome", "open_app", {"name": "chrome"}),
    ("launch Notepad", "open_app", {"name": "notepad"}),
    ("start Calculator", "open_app", {"name": "calculator"}),
    ("open Spotify", "open_app", {"name": "spotify"}),
    ("launch VS Code", "open_app", {"name": "vs code"}),
    ("hey jarvis could you open whatsapp", "open_app", {"name": "whatsapp"}),
    ("open File Explorer", "open_system_tool", {"tool": "file_explorer"}),
    ("open Task Manager", "open_system_tool", {"tool": "task_manager"}),
    ("open Settings", "open_system_tool", {"tool": "settings"}),
    ("close Chrome", "close_app", {"name": "chrome"}),
    ("quit Notepad", "close_app", {"name": "notepad"}),
    ("kill Spotify", "kill_process", {"target": "spotify"}),
    ("open youtube", "open_website", {"url": "https://www.youtube.com"}),
    ("navigate to https://github.com", "open_website", {"url": "https://github.com"}),
    ("go to website reddit.com", "open_website", {"url": "https://reddit.com"}),
    ("open url https://news.ycombinator.com", "open_website", {"url": "https://news.ycombinator.com"}),
    ("search google for fast python web frameworks", "search_in_browser", {"engine": "google", "query": "fast python web frameworks"}),
    ("search youtube for lofi beats to study", "search_in_browser", {"engine": "youtube", "query": "lofi beats to study"}),
    ("play despacito on youtube", "search_in_browser", {"engine": "youtube", "query": "despacito"}),
    ("what time is it", "get_time_date", {}),
    ("what's the date today", "get_time_date", {}),
    ("battery level", "system_status", {"focus": "battery"}),
    ("how much battery do i have", "system_status", {"focus": "battery"}),
    ("cpu usage", "system_status", {"focus": "cpu"}),
    ("next track", "media_control", {"action": "next"}),
    ("pause the music", "media_control", {"action": "play_pause"}),
    ("remember that my email is user@test.com", "memory_remember", {"fact": "my email is user@test.com"}),
    ("who am i", "memory_recall", {}),
    ("what do you know about me", "memory_recall", {}),
    ("forget my email", "memory_forget", {"topic": "my email"}),
    ("switch to gemini", "switch_model", {"model": "gemini"}),
    ("switch to ollama", "switch_model", {"model": "ollama"}),
    ("change model to groq", "switch_model", {"model": "groq"}),
    ("use openai", "switch_model", {"model": "openai"}),
    ("switch to gpt-4o", "switch_model", {"model": "gpt-4o"}),
    ("use claude", "switch_model", {"model": "claude"}),
    ("set model to mixtral", "switch_model", {"model": "mixtral"}),
    ("switch to llama-3.1-8b-instant", "switch_model", {"model": "llama-3.1-8b-instant"}),
    ("stop", "automation_control", {"action": "stop"}),
    ("emergency stop", "automation_control", {"action": "stop"}),
    ("resume automation", "automation_control", {"action": "resume"}),
    ("enable dry run mode", "automation_control", {"action": "dry_run_on"}),
    ("turn off dry run", "automation_control", {"action": "dry_run_off"}),
    ("start recording workflow morning setup", "workflow_record_start", {"name": "morning setup"}),
    ("stop recording", "workflow_record_stop", {}),
    ("replay send report with name=Rahul", "workflow_replay", {"name": "send report", "params": {"name": "Rahul"}}),
])
def test_fast_path_commands(query, tool, args):
    decision = router.fast_route(query)
    assert decision is not None, f"no fast route for {query!r}"
    assert decision.tool == tool, (query, decision)
    for k, v in args.items():
        assert decision.args.get(k) == v, (query, k, decision.args)


# ── Multi-step / UI commands go to the agent with action tools ─────────────────
@pytest.mark.parametrize("query", [
    "new tab in browser", "close current tab", "refresh the page", "go back in browser", "go forward in browser",
    "scroll down the page", "scroll up to top", "click the submit button", "click on login link",
    "fill in the form with username testuser", "create file notes.txt with hello world", "delete file temp.log",
    "open chrome and search for the latest ai news then open the first result",
    "send a whatsapp to mom saying I reached", "set a reminder for 6 pm", "organize my downloads folder",
])
def test_agent_commands_are_imperative(query):
    assert router.is_imperative(query)


# ── Questions about memory/vision/files are conversational (read-only tools only) ──
@pytest.mark.parametrize("query", [
    "what is my college name", "my favorite programming language is Python", "what is my name", "where do I live",
    "recall my project details", "what did I ask you yesterday", "what are my saved preferences",
    "describe what is on my screen", "what is on my screen right now", "where is the submit button on screen",
    "what is my current CPU and RAM usage", "what is the size of data folder", "check if file pyproject.toml exists",
])
def test_questions_are_not_imperative(query):
    assert not router.is_imperative(query)
    d = router.fast_route(query)
    assert d is None or d.tool in ("memory_recall", "system_status", "get_time_date", "list_windows")


def test_model_aliases():
    assert resolve_alias("groq") == "groq/llama-3.3-70b-versatile"
    assert resolve_alias("ollama vision").startswith("ollama/")
    assert resolve_alias("qwen 2b").startswith("ollama/")
    assert resolve_alias("gpt-4o") == "openai/gpt-4o"


@pytest.mark.parametrize("query,url_part", [
    ("search youtube for python tutorials", "youtube.com/results?search_query=python+tutorials"),
    ("google best laptops 2026", "google.com/search?q=best+laptops+2026"),
])
def test_engine_fast_plans_use_direct_urls(query, url_part):
    steps = task_planner.fast_plan(query)
    assert steps and steps[0]["action"] == "navigate" and url_part in steps[0]["target"]


def test_planner_drops_shell_steps():
    from automation.engine.planner import sanitize_steps

    steps = sanitize_steps([
        {"action": "run_command", "target": "shutdown /s /t 0"},
        {"action": "shell", "target": "del C:\\\\"},
        {"action": "open_app", "target": "notepad"},
    ])
    assert [s["action"] for s in steps] == ["open_app"]
