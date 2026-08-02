"""Unit tests for Jinja2 Prompt Manager."""

import pytest
from app.prompts.manager import prompt_manager
from app.utils.exceptions import PromptNotFoundError


def test_render_system_prompt():
    rendered = prompt_manager.render_prompt(
        "system/jarvis_system.jinja2",
        {"app_name": "JARVIS Test", "app_env": "test", "provider": "openai", "model": "gpt-4o"},
    )
    assert "JARVIS Test" in rendered
    assert "gpt-4o" in rendered


def test_render_developer_reasoning_prompt():
    rendered = prompt_manager.render_prompt(
        "developer/reasoning_developer.jinja2",
        {"user_query": "Build an AI system"},
    )
    assert "Build an AI system" in rendered
    assert "json" in rendered.lower()


def test_missing_prompt_raises_error():
    with pytest.raises(PromptNotFoundError):
        prompt_manager.render_prompt("non_existent_folder/missing.jinja2", {})
