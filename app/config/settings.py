"""Compatibility shim: API settings come from the single configuration in core.config."""

from core.config import BASE_DIR, LLMProviderType, Settings, settings

__all__ = ["BASE_DIR", "LLMProviderType", "Settings", "settings"]
