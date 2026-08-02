"""Domain & System Exceptions Hierarchy for JARVIS.

Provides custom exception classes to ensure predictable, structured error handling
across all layers of the application.
"""

from typing import Any, Optional, Dict


class JarvisError(Exception):
    """Base exception for all JARVIS system errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(JarvisError):
    """Raised when environment variables or system configurations are missing or invalid."""
    pass


class LLMError(JarvisError):
    """Base class for all LLM provider related errors."""
    pass


class LLMProviderNotFoundError(LLMError):
    """Raised when an unsupported or unconfigured LLM provider is requested."""
    pass


class LLMAuthenticationError(LLMError):
    """Raised when API key validation fails for an LLM provider."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when an LLM API request times out."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when an LLM API rate limit or quota is exceeded."""
    pass


class LLMAPIError(LLMError):
    """Raised when an external LLM API returns a non-200 HTTP response or payload error."""
    pass


class PromptNotFoundError(JarvisError):
    """Raised when a requested Jinja2 template or prompt file cannot be located."""
    pass


class IntentDetectionError(JarvisError):
    """Raised when intent parsing or classification fails."""
    pass


class ReasoningError(JarvisError):
    """Raised when reasoning or structured output generation encounters a failure."""
    pass


class ToolExecutionError(JarvisError):
    """Raised when execution of a tool fails."""
    pass


class DatabaseError(JarvisError):
    """Raised when SQLite or persistence layer errors occur."""
    pass
