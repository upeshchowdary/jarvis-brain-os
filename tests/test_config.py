"""Unit tests for Settings configuration loading."""

from app.config.settings import settings, LLMProviderType


def test_default_settings_values():
    assert settings.APP_NAME == "JARVIS Brain Engine"
    assert settings.DEFAULT_MODEL.split("/", 1)[0] in {p.value for p in LLMProviderType}
    assert settings.TEMPERATURE >= 0.0
    assert settings.MAX_TOKENS > 0
    assert settings.HOST == "127.0.0.1"


def test_api_key_resolver():
    openai_key = settings.get_api_key_for_provider(LLMProviderType.OPENAI)
    # Key may be None or string
    assert openai_key is None or isinstance(openai_key, str)
