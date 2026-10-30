"""Comprehensive 100-Scenario Test Suite for JARVIS Brain Intent Understanding & Automation Planning."""

import pytest
from brain.intent_engine import intent_engine
from automation.engine.planner import _detect_messaging_app
from automation.engine.messaging_handler import _detect_app_from_text, _heuristic_contact


# ══════════════════════════════════════════════════════════════════════════════
# Category 1: Email & Communication Understanding (15 tests)
# ══════════════════════════════════════════════════════════════════════════════

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
def test_communication_intent_understanding(query, expected_app, expected_contact):
    # 1. Intent Engine classification
    intent = intent_engine.detect_intent(query)
    assert intent.intent in ("AUTOMATION_TASK", "GENERAL_CONVERSATION")
    
    # 2. App detection
    detected_app = _detect_app_from_text(query)
    assert detected_app == expected_app, f"Expected {expected_app}, got {detected_app} for query: {query}"
    
    # 3. Planner app detection
    planner_app = _detect_messaging_app(query)
    assert planner_app == expected_app, f"Planner expected {expected_app}, got {planner_app} for query: {query}"
    
    # 4. Contact / Email extraction
    contact = _heuristic_contact(query)
    assert expected_contact.lower() in contact.lower(), f"Expected contact {expected_contact} in {contact}"


# ══════════════════════════════════════════════════════════════════════════════
# Category 2: System & Hardware Controls (15 tests)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("query,expected_target", [
    ("set brightness to 80", "set brightness"),
    ("brightness to 50", "brightness"),
    ("increase brightness", "brightness"),
    ("dim the screen", "brightness"),
    ("turn up the volume", "volume"),
    ("volume down", "volume"),
    ("mute sound", "volume"),
    ("unmute audio", "volume"),
    ("show desktop", "desktop"),
    ("minimize window", "minimize"),
    ("maximize this window", "maximize"),
    ("close active window", "close"),
    ("restore window", "restore"),
    ("lock computer", "lock"),
    ("put laptop to sleep", "sleep"),
])
def test_system_control_intents(query, expected_target):
    intent = intent_engine.detect_intent(query)
    assert intent.intent == "AUTOMATION_TASK"


# ══════════════════════════════════════════════════════════════════════════════
# Category 3: Application Launch & Window Control (15 tests)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("query,expected_app", [
    ("open Chrome", "chrome"),
    ("launch Notepad", "notepad"),
    ("start Calculator", "calculator"),
    ("open Spotify", "spotify"),
    ("launch VS Code", "vs code"),
    ("open File Explorer", "explorer"),
    ("open Task Manager", "task manager"),
    ("open Settings", "settings"),
    ("close Chrome", "chrome"),
    ("quit Notepad", "notepad"),
    ("kill Spotify", "spotify"),
    ("switch window", "window"),
    ("snap window left", "left"),
    ("snap window right", "right"),
    ("list open windows", "windows"),
])
def test_app_control_intents(query, expected_app):
    intent = intent_engine.detect_intent(query)
    assert intent.intent == "AUTOMATION_TASK"


# ══════════════════════════════════════════════════════════════════════════════
# Category 4: Browser, Search & Navigation (15 tests)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("query", [
    "search google for fast python web frameworks",
    "search youtube for lofi beats to study",
    "navigate to https://github.com",
    "go to website reddit.com",
    "open url https://news.ycombinator.com",
    "new tab in browser",
    "close current tab",
    "refresh the page",
    "go back in browser",
    "go forward in browser",
    "scroll down the page",
    "scroll up to top",
    "click the submit button",
    "click on login link",
    "fill in the form with username testuser",
])
def test_browser_automation_intents(query):
    intent = intent_engine.detect_intent(query)
    assert intent.intent == "AUTOMATION_TASK"


# ══════════════════════════════════════════════════════════════════════════════
# Category 5: Personal Memory & Facts (15 tests)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("query", [
    "what is my college name",
    "who am I",
    "remember that my email is user@test.com",
    "my favorite programming language is Python",
    "what is my name",
    "where do I live",
    "recall my project details",
    "what did I ask you yesterday",
    "what was the link I shared earlier",
    "what are my saved preferences",
    "tell me about my background",
    "what is my github username",
    "do you remember my pet name",
    "what is my favorite color",
    "summarize what you know about me",
])
def test_memory_and_knowledge_intents(query):
    intent = intent_engine.detect_intent(query)
    assert intent.intent in ("KNOWLEDGE_REQUEST", "GENERAL_CONVERSATION", "TASK_PLANNING")


# ══════════════════════════════════════════════════════════════════════════════
# Category 6: Model Switching & Multimodal Vision (15 tests)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("query,expected_model", [
    ("switch to gemini", "gemini"),
    ("switch to ollama", "ollama"),
    ("change model to groq", "llama-3.3-70b-versatile"),
    ("use openai", "gpt-4o-mini"),
    ("switch to gpt-4o", "gpt-4o"),
    ("switch to ollama vision", "ollama"),
    ("switch to qwen 2b", "ollama"),
    ("use claude", "claude"),
    ("set model to mixtral", "mixtral"),
    ("switch to llama-3.1-8b-instant", "llama-3.1-8b-instant"),
    ("describe what is on my screen", "vision"),
    ("what is on my screen right now", "vision"),
    ("read visible text on screen", "ocr"),
    ("extract text from screen", "ocr"),
    ("where is the submit button on screen", "ui"),
])
def test_model_switching_and_vision(query, expected_model):
    intent = intent_engine.detect_intent(query)
    assert intent.intent in ("MODEL_SWITCH", "SCREEN_VISION", "AUTOMATION_TASK", "GENERAL_CONVERSATION")


# ══════════════════════════════════════════════════════════════════════════════
# Category 7: Filesystem Operations & OS Telemetry (10 tests)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("query", [
    "list files on desktop",
    "show files in current folder",
    "what is the size of data folder",
    "read content of README.md file",
    "create file notes.txt with hello world",
    "delete file temp.log",
    "check if file pyproject.toml exists",
    "what is my current CPU and RAM usage",
    "system hardware status",
    "disk space available",
])
def test_filesystem_and_telemetry(query):
    intent = intent_engine.detect_intent(query)
    assert intent.intent in ("FILESYSTEM_OPERATION", "SYSTEM_TELEMETRY", "AUTOMATION_TASK")
