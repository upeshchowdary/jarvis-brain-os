"""Configuration Management Module for JARVIS AI Operating System.

This module defines system settings using Pydantic BaseSettings, which automatically
loads configuration from environment variables or .env files with strict type validation.
"""

from enum import Enum
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class LLMProviderType(str, Enum):
    OPENAI = "openai"
    GROQ = "groq"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    DEEPSEEK = "deepseek"
    LMSTUDIO = "lmstudio"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # Server Info
    APP_NAME: str = "JARVIS Brain Engine"
    APP_ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # LLM Settings
    LLM_PROVIDER: LLMProviderType = Field(default=LLMProviderType.OPENAI)
    MODEL_NAME: str = Field(default="gpt-4o")
    LLM_TEMPERATURE: float = Field(default=0.2, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=2048, ge=1)
    LLM_TIMEOUT_SECONDS: int = Field(default=60, ge=1)

    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None

    # Local LLM Endpoints
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LM_STUDIO_BASE_URL: str = "http://localhost:1234/v1"

    # Paths & Persistence
    LOG_LEVEL: str = "INFO"
    LOG_FILE_PATH: str = "logs/jarvis.log"
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/jarvis.db"

    @property
    def base_dir(self) -> Path:
        return BASE_DIR

    def get_api_key_for_provider(self, provider: LLMProviderType) -> Optional[str]:
        """Retrieve API key corresponding to the specified provider."""
        mapping = {
            LLMProviderType.OPENAI: self.OPENAI_API_KEY,
            LLMProviderType.GROQ: self.GROQ_API_KEY,
            LLMProviderType.ANTHROPIC: self.ANTHROPIC_API_KEY,
            LLMProviderType.GEMINI: self.GEMINI_API_KEY,
            LLMProviderType.OPENROUTER: self.OPENROUTER_API_KEY,
            LLMProviderType.DEEPSEEK: self.DEEPSEEK_API_KEY,
        }
        return mapping.get(provider)


# Global settings singleton
settings = Settings()
