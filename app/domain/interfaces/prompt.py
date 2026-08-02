"""Abstract Interface for Prompt Management Engine."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BasePromptManager(ABC):
    """Abstract interface for template loading and compilation."""

    @abstractmethod
    def render_prompt(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render a specified Jinja2 prompt template with the provided variable context."""
        pass
