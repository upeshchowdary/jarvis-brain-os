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


def test_plan_b_defaults():
    """G2: Groq gpt-oss for the intent step (Gemini Flash behind it), Gemini Flash for chat and actions."""
    from core.config import Settings

    fields = Settings.model_fields
    assert fields["INTENT_MODEL"].default == "groq/openai/gpt-oss-120b"
    assert fields["INTENT_FALLBACK_MODEL"].default == "gemini/gemini-3.5-flash"
    for name in ("DEFAULT_MODEL", "FAST_MODEL", "ACTION_MODEL"):
        assert fields[name].default == "gemini/gemini-3.5-flash", name
    assert fields["LLM_FALLBACKS"].default_factory() == ["groq/openai/gpt-oss-120b"]
    effort = fields["REASONING_EFFORT"].default_factory()
    assert effort["gemini-3.5-flash"] == "minimal" and effort["openai/gpt-oss-120b"] == "low"


def test_env_example_matches_the_defaults():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")
    lines = {ln.split("=", 1)[0]: ln.split("=", 1)[1].split("#")[0].strip()
             for ln in text.splitlines() if "=" in ln and not ln.startswith("#")}
    assert lines["INTENT_MODEL"] == "groq/openai/gpt-oss-120b"
    assert lines["INTENT_FALLBACK_MODEL"] == "gemini/gemini-3.5-flash"
    for name in ("DEFAULT_MODEL", "FAST_MODEL", "ACTION_MODEL"):
        assert lines[name] == "gemini/gemini-3.5-flash", name
    assert lines["LLM_FALLBACKS"] == "groq/openai/gpt-oss-120b"
    assert "GROQ_API_KEYS" in lines and "per Groq organisation" in text
