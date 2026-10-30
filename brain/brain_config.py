"""Compatibility shim: the brain shares the single JARVIS configuration in core.config."""

from core.config import BASE_DIR, PersonalityType, Settings as BrainConfig, settings as brain_config

__all__ = ["BASE_DIR", "BrainConfig", "PersonalityType", "brain_config"]
