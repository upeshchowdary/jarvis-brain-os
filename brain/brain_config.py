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

    # API Keys Configuration
    GROQ_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    DEFAULT_MODEL: str = "llama-3.3-70b-versatile"
    FAST_MODEL: str = "llama-3.1-8b-instant"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "qwen3.5:latest"
    OLLAMA_VISION_MODEL: str = "qwen3-vl:2b"
    GROQ_VISION_MODEL: str = "llama-3.2-90b-vision-preview"
    CLAUDE_VISION_MODEL: str = "claude-3-5-sonnet-20241022"
    VISION_PRIMARY_PROVIDER: str = "gemini"   # gemini | groq | ollama
    ENABLE_HYBRID_ROUTING: bool = True

    # Vision-specific configuration (separate from general LLM timeout)
    VISION_SERVER_HOST: str = "127.0.0.1"
    VISION_SERVER_PORT: int = 8001
    VISION_ENABLED: bool = True
    VISION_CAPTURE_FPS: float = Field(default=2.0, ge=0.1, le=30.0)
    VISION_STATE_MAX_AGE: float = Field(default=15.0, ge=1.0)
    VISION_CHANGE_THRESHOLD: float = Field(default=2.0, ge=0.0)
    VISION_TIMEOUT: float = Field(default=45.0, ge=10.0)
    VISION_IMAGE_SIZE: int = Field(default=768, ge=64, le=2048)
    VISION_CACHE_TTL: int = Field(default=10, ge=0)
    VISION_RACE_PROVIDERS: bool = True   # fire Gemini + Groq simultaneously
    VISION_QUICK_MODE_MS: int = Field(default=300, ge=50)  # target latency for quick mode

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
