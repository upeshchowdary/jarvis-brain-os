"""Personality Manager for managing system personas and system prompt style injection."""

from typing import Dict, Any, Optional
from brain.brain_config import brain_config, PersonalityType
from brain.logger import logger


class PersonalityManager:
    """Manages system personas and injected system instructions."""

    def __init__(self) -> None:
        self._current_personality: PersonalityType = brain_config.DEFAULT_PERSONALITY
        self._custom_personas: Dict[str, str] = {}

        self._builtin_prompts: Dict[PersonalityType, str] = {
            PersonalityType.ASSISTANT: (
                "You are JARVIS, an advanced, highly intelligent AI Operating Assistant. "
                "Your tone is polite, executive, precise, and proactive. "
                "Provide helpful, clear, and structured answers while maintaining an efficient assistant persona."
            ),
            PersonalityType.PROFESSIONAL: (
                "You are a Senior Corporate Executive AI Advisor. "
                "Maintain a formal, professional, clear, and concise business tone. "
                "Avoid casual humor and focus on precision and action-oriented results."
            ),
            PersonalityType.FRIENDLY: (
                "You are a friendly, warm, and empathetic AI Companion. "
                "Speak conversationally, encourage the user, and maintain an approachable, supportive tone."
            ),
            PersonalityType.DEVELOPER: (
                "You are a Principal Software Engineer and System Architect. "
                "Provide deep technical rigor, clean code snippets, design patterns, "
                "and precise architectural analysis."
            ),
            PersonalityType.MINIMAL: (
                "You are a high-efficiency minimal AI assistant. "
                "Give bulleted, highly concise answers without filler or fluff."
            ),
        }

    def set_personality(self, personality: PersonalityType | str) -> bool:
        """Set the active system personality."""
        try:
            if isinstance(personality, str):
                personality = PersonalityType(personality.lower())
            self._current_personality = personality
            logger.info(f"Set active brain personality to: '{self._current_personality.value}'")
            return True
        except ValueError:
            logger.warning(f"Unsupported personality requested: '{personality}'")
            return False

    def register_custom_personality(self, name: str, system_prompt: str) -> None:
        """Register a custom system persona at runtime."""
        self._custom_personas[name.lower()] = system_prompt
        logger.info(f"Registered custom personality: '{name}'")

    def get_system_prompt(self, personality_override: Optional[str] = None) -> str:
        """Retrieve system prompt text for active or overridden personality."""
        target_name = (personality_override or self._current_personality.value).lower()

        if target_name in self._custom_personas:
            return self._custom_personas[target_name]

        try:
            p_enum = PersonalityType(target_name)
            return self._builtin_prompts[p_enum]
        except ValueError:
            return self._builtin_prompts[PersonalityType.ASSISTANT]


# Global personality manager singleton
personality_manager = PersonalityManager()
