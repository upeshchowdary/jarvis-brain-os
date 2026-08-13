"""JARVIS Visual Reasoner v3 — Unified Vision Provider Orchestrator.

Provides a single interface for visual reasoning tasks with automatic
fallback queues across vision model providers.

v3: Removed deleted providers (groq.py, qwen.py, llava.py).
    Uses connection-pooled providers from v3 modules.
"""

from typing import Dict, Any, List, Optional
from loguru import logger

from brain.brain_config import brain_config
from vision.providers.base import BaseVisionProvider
from vision.providers.openai import OpenAIVisionProvider
from vision.providers.gemini import GeminiVisionProvider
from vision.providers.groq_vision import GroqVisionProvider
from vision.providers.ollama_vision import OllamaVisionProvider


class VisualReasoner:
    """Central abstraction orchestrator for vision provider models."""

    def __init__(self) -> None:
        self._providers: Dict[str, BaseVisionProvider] = {}
        self.active_provider_name: str = "gemini"

        # Initialize providers from environment
        if brain_config.GEMINI_API_KEY:
            self.register_provider("gemini", GeminiVisionProvider(api_key=brain_config.GEMINI_API_KEY))
        if brain_config.OPENAI_API_KEY:
            self.register_provider("openai", OpenAIVisionProvider(api_key=brain_config.OPENAI_API_KEY))
        if brain_config.GROQ_API_KEY:
            self.register_provider("groq", GroqVisionProvider(api_key=brain_config.GROQ_API_KEY))

        # Ollama local (always available as fallback)
        self.register_provider("ollama", OllamaVisionProvider())

        # Select initial active provider
        if "gemini" in self._providers:
            self.active_provider_name = "gemini"
        elif "openai" in self._providers:
            self.active_provider_name = "openai"
        elif "groq" in self._providers:
            self.active_provider_name = "groq"
        else:
            self.active_provider_name = "ollama"

    def register_provider(self, name: str, provider: BaseVisionProvider) -> None:
        """Dynamically register or overwrite a vision provider adapter."""
        self._providers[name.lower()] = provider
        logger.info(f"VisualReasoner registered provider: '{name.lower()}'")

    def switch_provider(self, name: str) -> bool:
        """Switch active vision provider."""
        key = name.lower()
        if key in self._providers:
            self.active_provider_name = key
            logger.info(f"VisualReasoner switched to: '{key}'")
            return True
        logger.warning(f"VisualReasoner provider '{key}' is not registered.")
        return False

    def list_providers(self) -> List[str]:
        """Returns names of all registered vision providers."""
        return list(self._providers.keys())

    async def analyze_image(
        self,
        image: Any,
        prompt: str = "Describe this visual frame in detail.",
        preferred_provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Submit an image for visual analysis with automatic provider fallback queue."""
        if image is None:
            return {"success": False, "error": "No image payload provided."}

        # Build candidate queue
        queue = []
        target = (preferred_provider or self.active_provider_name).lower()
        if target in self._providers:
            queue.append(target)

        for p_name in self._providers:
            if p_name not in queue:
                queue.append(p_name)

        last_error = None
        for p_name in queue:
            provider = self._providers[p_name]
            logger.info(f"VisualReasoner trying provider '{p_name}'...")
            res = await provider.analyze_image(image=image, prompt=prompt, **kwargs)
            if res.get("success"):
                return res
            logger.warning(f"Provider '{p_name}' failed: {res.get('error')}. Next...")
            last_error = res.get("error")

        return {
            "success": False,
            "error": f"All vision providers failed. Last error: {last_error}",
            "analysis": "Visual analysis unavailable across vision providers.",
        }

    async def describe_scene(self, image: Any) -> str:
        """Produce a concise natural language description of a scene/screen."""
        prompt = (
            "Analyze this image and describe the overall scene, active applications, "
            "visual layout, dominant features, lighting/colors, and any prominent text."
        )
        res = await self.analyze_image(image, prompt=prompt)
        return res.get("analysis", "No scene description available.")


# Global singleton instance
visual_reasoner = VisualReasoner()
