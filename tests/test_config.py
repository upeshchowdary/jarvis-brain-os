"""Unit tests for Settings configuration loading."""

import pytest

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


async def test_tests_never_reach_a_real_local_model():
    """A local Ollama / LM Studio may be running on the dev PC: the suite points them at a closed port."""
    import time

    from core.config import settings
    from core.llm import LLMUnavailable, llm

    assert settings.OLLAMA_BASE_URL == "http://127.0.0.1:9"
    assert settings.LM_STUDIO_BASE_URL == "http://127.0.0.1:9/v1"
    started = time.perf_counter()
    with pytest.raises(LLMUnavailable):
        await llm.chat([{"role": "user", "content": "x"}], model="ollama/qwen3.5:latest", fallbacks=False, retries=0)
    assert time.perf_counter() - started < 3.0
