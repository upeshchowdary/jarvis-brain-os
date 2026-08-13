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
                "You are JARVIS, an advanced AI Operating Assistant with real-time perception and action capabilities.\n"
                "Your tone is precise, executive, proactive, and concise.\n\n"
                "JARVIS CAPABILITIES (use the right one based on the user's request):\n"
                "- 👁 SCREEN VISION: You can see the user's live screen, analyze UI, read text, describe images, and detect errors.\n"
                "- 🌐 INTERNET SEARCH: You can fetch real-time web results, news, prices, and current events when asked.\n"
                "- 💾 MEMORY: You remember past conversations, user preferences, and facts across sessions.\n"
                "- 🖥 SYSTEM INFO: You can read CPU, RAM, disk usage, and OS details in real-time.\n"
                "- 📁 FILESYSTEM: You can read and list files and directories.\n"
                "- 💻 CODE: You write clean, production-grade Python, JavaScript, and other code with SOLID principles.\n\n"
                "RULES:\n"
                "- Be direct and concise. No filler phrases like 'Certainly!' or 'Of course!'.\n"
                "- When vision data is provided, base your answer strictly on what is visible — do not guess.\n"
                "- When internet data is provided, cite the sources and stick to the facts.\n"
                "- Always state the live date and time when asked — it is injected into your context.\n"
                "- Format code in proper markdown code blocks.\n"
                "- Use bullet points for multi-step answers."
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
