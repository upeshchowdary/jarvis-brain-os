"""Unit tests for LLM Abstraction Factory."""

import pytest
from app.llm.factory import LLMFactory
from app.llm.providers import OpenAIProvider, GroqProvider, OllamaProvider
from app.utils.exceptions import LLMProviderNotFoundError


def test_factory_creates_openai_provider():
    provider = LLMFactory.create_provider(provider_name="openai", model_name="gpt-4o", api_key="dummy_key")
    assert isinstance(provider, OpenAIProvider)
    assert provider.model_name == "gpt-4o"


def test_factory_creates_groq_provider():
    provider = LLMFactory.create_provider(provider_name="groq", api_key="dummy_key")
    assert isinstance(provider, GroqProvider)


def test_factory_creates_ollama_provider():
    provider = LLMFactory.create_provider(provider_name="ollama")
    assert isinstance(provider, OllamaProvider)


def test_factory_invalid_provider_raises_exception():
    with pytest.raises(LLMProviderNotFoundError):
        LLMFactory.create_provider(provider_name="non_existent_provider")
