"""Unit tests for the modular JARVIS Brain Framework."""

import pytest
from brain.brain_config import brain_config, PersonalityType
from brain.model_manager import model_manager
from brain.personality_manager import personality_manager
from brain.prompt_manager import prompt_manager
from brain.context_manager import context_manager
from brain.conversation_manager import conversation_manager
from brain.intent_engine import intent_engine
from brain.reasoning_engine import reasoning_engine
from brain.planner import planner
from brain.reflection_engine import reflection_engine
from brain.tool_router import tool_router, BaseBrainTool
from brain.knowledge_manager import knowledge_manager
from brain.brain_manager import brain_manager


def test_brain_config():
    assert brain_config.SYSTEM_NAME == "JARVIS"
    assert brain_config.DEFAULT_MODEL == "llama-3.3-70b-versatile"


def test_model_manager_list():
    models = model_manager.list_models()
    assert "llama-3.3-70b-versatile" in models


def test_personality_manager():
    personality_manager.set_personality(PersonalityType.DEVELOPER)
    prompt = personality_manager.get_system_prompt()
    assert "Software Engineer" in prompt
    # Reset back to assistant
    personality_manager.set_personality(PersonalityType.ASSISTANT)


def test_prompt_manager_render():
    rendered = prompt_manager.render("chat", {
        "personality_prompt": "Test Persona",
        "current_time": "12:00",
        "active_model": "llama-3.3",
        "user_query": "Hello",
    })
    assert "Test Persona" in rendered
    assert "Hello" in rendered


def test_context_manager():
    ctx = context_manager.build_context("Test Query")
    assert ctx["user_query"] == "Test Query"
    assert "current_date" in ctx


def test_conversation_manager():
    sid = "test_session_xyz"
    conversation_manager.add_user_message(sid, "Hello")
    history = conversation_manager.get_history(sid)
    assert len(history) == 1
    assert history[0]["content"] == "Hello"
    conversation_manager.clear_session(sid)


def test_intent_engine():
    intent = intent_engine.detect_intent("Can you open Chrome?")
    assert intent.intent == "OPEN_APPLICATION"
    assert intent.arguments.get("application_name") == "chrome"


def test_planner():
    plan = planner.create_plan("Build a website")
    assert plan.complexity == "high"
    assert len(plan.steps) >= 3


def test_reflection_engine():
    ref = reflection_engine.evaluate_response("Query", "This is a comprehensive response answer.")
    assert ref.correctness_score >= 0.90
    assert ref.answered_correctly is True


def test_tool_router_discovery():
    tools = tool_router.discover_tools()
    assert isinstance(tools, list)


@pytest.mark.asyncio
async def test_brain_manager_execution():
    output = await brain_manager.execute_cognitive_pipeline(
        user_query="Hello JARVIS!",
        session_id="test_pipeline_session",
    )
    assert output.intent.intent == "GENERAL_CONVERSATION"
    assert len(output.response) > 0
    assert output.reflection.correctness_score > 0.0
