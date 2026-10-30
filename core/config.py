"""Single source of configuration for JARVIS (replaces app/config + brain/brain_config).

Values come from environment variables or the project ``.env`` file. Secrets are
never hard-coded; see ``.env.example`` for every supported key.
"""

from __future__ import annotations

import os
import secrets
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from core.paths import BASE_DIR

ENV_FILE = Path(os.environ.get("JARVIS_ENV_FILE") or (BASE_DIR / ".env"))

# List settings accept JSON (["a","b"]) or a plain comma-separated string (a,b).
StrList = Annotated[List[str], NoDecode]


class PersonalityType(str, Enum):
    ASSISTANT = "assistant"
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    DEVELOPER = "developer"
    MINIMAL = "minimal"


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
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Server ──────────────────────────────────────────────────────────
    APP_NAME: str = "JARVIS Brain Engine"
    SYSTEM_NAME: str = "JARVIS"
    SYSTEM_VERSION: str = "3.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = False
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # ── API security ────────────────────────────────────────────────────
    JARVIS_API_TOKEN: Optional[str] = None
    CORS_ORIGINS: StrList = Field(default_factory=list)
    RATE_LIMIT_PER_MINUTE: int = Field(default=120, ge=1)

    # ── LLM ─────────────────────────────────────────────────────────────
    # Models are written "provider/model". A bare name is mapped to a provider by prefix.
    DEFAULT_MODEL: str = "gemini/gemini-3.5-flash-lite"
    FAST_MODEL: str = "gemini/gemini-3.5-flash-lite"
    LLM_FALLBACKS: StrList = Field(
        default_factory=lambda: ["groq/llama-3.3-70b-versatile", "openai/gpt-4o-mini", "ollama/qwen3.5:latest"]
    )
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"
    TEMPERATURE: float = Field(default=0.2, ge=0.0, le=2.0)
    MAX_TOKENS: int = Field(default=1024, ge=1)
    TIMEOUT_SECONDS: float = Field(default=30.0, ge=1.0)
    CONNECT_TIMEOUT_SECONDS: float = Field(default=4.0, ge=0.5)
    MAX_RETRIES: int = Field(default=2, ge=0)

    GROQ_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None

    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    LM_STUDIO_BASE_URL: str = "http://127.0.0.1:1234/v1"
    OLLAMA_MODEL: str = "qwen3.5:latest"

    # ── Privacy ─────────────────────────────────────────────────────────
    # "cloud": any configured provider may see file/screen content.
    # "local": file contents and screenshots are only sent to Ollama / LM Studio.
    PRIVACY_MODE: str = "cloud"

    # ── Safety ──────────────────────────────────────────────────────────
    DRY_RUN: bool = False
    CONFIRM_MEDIUM_RISK: bool = True
    CONFIRMATION_TTL_SECONDS: float = Field(default=60.0, ge=5.0)
    POWER_ACTION_DELAY_SECONDS: int = Field(default=30, ge=5)
    PROTECTED_PROCESSES: StrList = Field(default_factory=lambda: [
        "system", "system idle process", "registry", "smss.exe", "csrss.exe", "wininit.exe",
        "winlogon.exe", "services.exe", "lsass.exe", "svchost.exe", "explorer.exe", "dwm.exe",
        "fontdrvhost.exe", "sihost.exe", "ctfmon.exe", "audiodg.exe", "msmpeng.exe", "init", "systemd",
    ])

    # ── Router / agent ──────────────────────────────────────────────────
    ROUTER_USE_LLM: bool = True
    AGENT_MAX_STEPS: int = Field(default=5, ge=1, le=15)
    MAX_HISTORY_MESSAGES: int = 20
    MAX_CONVERSATION_TOKENS: int = 4096
    MAX_FACTS_PER_TURN: int = 12
    MEMORY_RECALL_MIN_SCORE: float = 0.35
    SESSION_TTL_SECONDS: float = 6 * 3600
    MAX_SESSIONS: int = 200
    DEFAULT_PERSONALITY: PersonalityType = PersonalityType.ASSISTANT

    # ── Background services (scheduler, health monitor, clipboard history) ──
    BACKGROUND_SERVICES: bool = True
    HEALTH_CPU_ALERT: float = 92.0
    HEALTH_RAM_ALERT: float = 92.0
    HEALTH_DISK_ALERT: float = 95.0
    HEALTH_BATTERY_ALERT: float = 15.0
    CLIPBOARD_HISTORY_SIZE: int = 50

    # ── Vision ──────────────────────────────────────────────────────────
    OLLAMA_VISION_MODEL: str = "qwen3-vl:2b"
    GROQ_VISION_MODEL: str = "llama-3.2-90b-vision-preview"
    CLAUDE_VISION_MODEL: str = "claude-3-5-sonnet-20241022"
    VISION_PRIMARY_PROVIDER: str = "gemini"
    VISION_SERVER_HOST: str = "127.0.0.1"
    VISION_SERVER_PORT: int = 8001
    VISION_ENABLED: bool = True
    VISION_CAPTURE_FPS: float = Field(default=2.0, ge=0.1, le=30.0)
    VISION_STATE_MAX_AGE: float = Field(default=15.0, ge=1.0)
    VISION_CHANGE_THRESHOLD: float = Field(default=2.0, ge=0.0)
    VISION_TIMEOUT: float = Field(default=45.0, ge=10.0)
    VISION_IMAGE_SIZE: int = Field(default=768, ge=64, le=2048)
    VISION_CACHE_TTL: int = Field(default=10, ge=0)
    VISION_RACE_PROVIDERS: bool = False
    VISION_QUICK_MODE_MS: int = Field(default=300, ge=50)

    # ── Logging ─────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE_PATH: str = "logs/jarvis.log"

    @field_validator("CORS_ORIGINS", "LLM_FALLBACKS", "PROTECTED_PROCESSES", mode="before")
    @classmethod
    def _split_list(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                import json
                return json.loads(v)
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("PRIVACY_MODE")
    @classmethod
    def _privacy(cls, v: str) -> str:
        v = (v or "cloud").strip().lower()
        if v not in ("cloud", "local"):
            raise ValueError("PRIVACY_MODE must be 'cloud' or 'local'")
        return v

    @property
    def base_dir(self) -> Path:
        return BASE_DIR

    @property
    def PROMPTS_DIR(self) -> Path:  # noqa: N802 - kept for compatibility
        return BASE_DIR / "brain" / "prompts_store"

    def api_key_for(self, provider: str) -> Optional[str]:
        return {
            "openai": self.OPENAI_API_KEY,
            "groq": self.GROQ_API_KEY,
            "anthropic": self.ANTHROPIC_API_KEY,
            "gemini": self.GEMINI_API_KEY,
            "openrouter": self.OPENROUTER_API_KEY,
            "deepseek": self.DEEPSEEK_API_KEY,
        }.get(provider)

    def get_api_key_for_provider(self, provider: LLMProviderType | str) -> Optional[str]:
        return self.api_key_for(provider.value if isinstance(provider, LLMProviderType) else provider)


settings = Settings()


def ensure_api_token(env_file: Path = ENV_FILE) -> str:
    """Return the API token, generating one into .env on first run."""
    if settings.JARVIS_API_TOKEN:
        return settings.JARVIS_API_TOKEN
    token = secrets.token_urlsafe(32)
    try:
        existing = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
        sep = "" if (not existing or existing.endswith("\n")) else "\n"
        env_file.write_text(f"{existing}{sep}JARVIS_API_TOKEN={token}\n", encoding="utf-8")
    except OSError:
        pass  # Read-only checkout: token still works for this process.
    settings.JARVIS_API_TOKEN = token
    return token
