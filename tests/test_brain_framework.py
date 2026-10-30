"""Unit tests for the brain framework (config, personality, conversation, router basics)."""

from brain.brain_config import PersonalityType, brain_config
from brain.conversation_manager import conversation_manager
from brain.personality_manager import personality_manager
from brain import router


def test_brain_config():
    assert brain_config.SYSTEM_NAME == "JARVIS"
    assert brain_config.DEFAULT_MODEL == "gemini/gemini-3.5-flash-lite"


def test_personality_manager():
    personality_manager.set_personality(PersonalityType.DEVELOPER)
    assert "Software Engineer" in personality_manager.get_system_prompt()
    personality_manager.set_personality(PersonalityType.ASSISTANT)
    assert "JARVIS" in personality_manager.get_system_prompt()


def test_conversation_manager():
    sid = "test_session_xyz"
    conversation_manager.add_user_message(sid, "Hello")
    history = conversation_manager.get_history(sid)
    assert len(history) == 1 and history[0]["content"] == "Hello"
    conversation_manager.clear_session(sid)


def test_router_open_and_switch():
    d = router.fast_route("Can you open Chrome?")
    assert d is not None and d.tool == "open_app" and d.args["name"] == "chrome"
    d = router.fast_route("launch Notepad")
    assert d.tool == "open_app" and d.args["name"] == "notepad"
    d = router.fast_route("switch to ollama")
    assert d.tool == "switch_model" and d.args["model"] == "ollama"


async def test_past_conversations_search_and_fact_extraction():
    from app.database.connection import db_manager
    from memory.long_term import long_term_memory

    extracted = await long_term_memory.auto_extract_facts("My name is Upesh and I am working on project JARVIS")
    assert len(extracted) >= 1
    assert (await long_term_memory.get_fact("user_name")).lower() == "upesh"
    assert isinstance(await db_manager.search_past_conversations("JARVIS", limit=5), list)
