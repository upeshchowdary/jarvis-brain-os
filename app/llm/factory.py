"""Dynamic LLM Provider Factory."""

from typing import Optional, Dict, Type, Any
from app.config.settings import settings, LLMProviderType
from app.domain.interfaces.llm import BaseLLMProvider
from app.llm.providers import (
    OpenAIProvider,
    GroqProvider,
    AnthropicProvider,
    GeminiProvider,
    OpenRouterProvider,
    OllamaProvider,
    DeepSeekProvider,
)
from app.utils.exceptions import LLMProviderNotFoundError, LLMAuthenticationError
from app.utils.logger import logger

PROVIDER_REGISTRY: Dict[str, Type[BaseLLMProvider]] = {
    LLMProviderType.OPENAI.value: OpenAIProvider,
    LLMProviderType.GROQ.value: GroqProvider,
    LLMProviderType.ANTHROPIC.value: AnthropicProvider,
    LLMProviderType.GEMINI.value: GeminiProvider,
    LLMProviderType.OPENROUTER.value: OpenRouterProvider,
    LLMProviderType.OLLAMA.value: OllamaProvider,
    LLMProviderType.DEEPSEEK.value: DeepSeekProvider,
    LLMProviderType.LMSTUDIO.value: OllamaProvider,  # Uses OpenAI-compatible HTTP interface
}


class LLMFactory:
    """Factory class to construct and instantiate LLM provider adapters at runtime."""

    @staticmethod
    def create_provider(
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseLLMProvider:
        """Instantiate specified LLM provider with fallback to global application settings."""
        target_provider_str = (provider_name or settings.LLM_PROVIDER.value).lower().strip()
        target_model = model_name or settings.MODEL_NAME

        if target_provider_str not in PROVIDER_REGISTRY:
            logger.error(f"Unsupported LLM provider requested: '{target_provider_str}'")
            raise LLMProviderNotFoundError(
                f"Provider '{target_provider_str}' is not supported. "
                f"Available providers: {list(PROVIDER_REGISTRY.keys())}"
            )

        provider_cls = PROVIDER_REGISTRY[target_provider_str]

        # Resolve API Key
        try:
            enum_provider = LLMProviderType(target_provider_str)
            target_api_key = api_key or settings.get_api_key_for_provider(enum_provider)
        except ValueError:
            target_api_key = api_key

        # Local providers do not strictly mandate API key
        if not target_api_key and target_provider_str not in (LLMProviderType.OLLAMA.value, LLMProviderType.LMSTUDIO.value):
            logger.warning(f"No API key provided or found in environment for provider '{target_provider_str}'")

        logger.info(f"Instantiating LLM Provider adapter: {target_provider_str} (Model: {target_model})")
        return provider_cls(
            api_key=target_api_key or "",
            model_name=target_model,
            timeout=float(settings.LLM_TIMEOUT_SECONDS),
            **kwargs,
        )
