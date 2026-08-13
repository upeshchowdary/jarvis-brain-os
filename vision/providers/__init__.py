"""JARVIS Vision Providers Package v3.

Active providers:
  - GeminiVisionProvider   — Google Gemini 2.5 Flash (PRIMARY)
  - GroqVisionProvider     — Groq Llama 4 Scout (SECONDARY)
  - OpenAIVisionProvider   — OpenAI GPT-4o-mini (TERTIARY)
  - OllamaVisionProvider   — Local models via Ollama (FALLBACK)

Removed in v3 (consolidated into OllamaVisionProvider):
  - groq.py (text-only adapter, not needed in vision pipeline)
  - llava.py (LLaVA is just another Ollama model)
  - qwen.py (Qwen-VL is just another Ollama model)
"""

from vision.providers.base import BaseVisionProvider
from vision.providers.gemini import GeminiVisionProvider
from vision.providers.groq_vision import GroqVisionProvider
from vision.providers.openai import OpenAIVisionProvider
from vision.providers.ollama_vision import OllamaVisionProvider

__all__ = [
    "BaseVisionProvider",
    "GeminiVisionProvider",
    "GroqVisionProvider",
    "OpenAIVisionProvider",
    "OllamaVisionProvider",
]
