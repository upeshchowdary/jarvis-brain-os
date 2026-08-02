"""Brain Framework Central Configuration Management."""

import os
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class PersonalityType(str, Enum):
    ASSISTANT = "assistant"
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    DEVELOPER = "developer"
    MINIMAL = "minimal"


class BrainConfig(BaseSettings):
    """Centralized configuration for JARVIS Brain Framework."""
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # Core System Metadata
    SYSTEM_NAME: str = "JARVIS"
    SYSTEM_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Groq API Configuration
    GROQ_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    DEFAULT_MODEL: str = "llama-3.3-70b-versatile"
    FAST_MODEL: str = "llama-3.1-8b-instant"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5"

    # LLM Hyperparameters
    TEMPERATURE: float = Field(default=0.2, ge=0.0, le=2.0)
    MAX_TOKENS: int = Field(default=2048, ge=1)
    TIMEOUT_SECONDS: float = Field(default=60.0, ge=1.0)
    MAX_RETRIES: int = Field(default=3, ge=0)

    # Personality & Conversation Memory Limits
    DEFAULT_PERSONALITY: PersonalityType = PersonalityType.ASSISTANT
    MAX_CONVERSATION_TOKENS: int = 4096
    MAX_HISTORY_MESSAGES: int = 20

    # Paths & Log Config
    LOG_LEVEL: str = "INFO"
    LOG_FILE_PATH: str = "logs/brain.log"
    PROMPTS_DIR: Path = Field(default_factory=lambda: BASE_DIR / "brain" / "prompts_store")

    @property
    def base_dir(self) -> Path:
        return BASE_DIR


# Global configuration singleton
brain_config = BrainConfig()
