"""Unit tests for Settings configuration loading."""

from app.config.settings import settings, LLMProviderType


def test_default_settings_values():
    assert settings.APP_NAME == "JARVIS Brain Engine"
    assert settings.LLM_PROVIDER in LLMProviderType
    assert settings.LLM_TEMPERATURE >= 0.0
    assert settings.LLM_MAX_TOKENS > 0


def test_api_key_resolver():
    openai_key = settings.get_api_key_for_provider(LLMProviderType.OPENAI)
    # Key may be None or string
    assert openai_key is None or isinstance(openai_key, str)
