"""Non-secret runtime configuration (reports the provider/model chat actually uses)."""

from typing import Any, Dict

from fastapi import APIRouter

from core.config import settings
from core.llm import llm, resolve_model

router = APIRouter(tags=["Configuration"])


@router.get("/config")
async def get_configuration() -> Dict[str, Any]:
    provider, model = resolve_model(settings.DEFAULT_MODEL)
    return {
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "debug_mode": settings.DEBUG,
        "default_llm_provider": provider,
        "default_model": model,
        "fast_model": "/".join(resolve_model(settings.FAST_MODEL)),
        "fallback_chain": settings.LLM_FALLBACKS,
        "configured_models": llm.configured_models(),
        "temperature": settings.TEMPERATURE,
        "max_tokens": settings.MAX_TOKENS,
        "timeout_seconds": settings.TIMEOUT_SECONDS,
        "privacy_mode": settings.PRIVACY_MODE,
        "dry_run": settings.DRY_RUN,
        "confirm_medium_risk": settings.CONFIRM_MEDIUM_RISK,
        "ollama_base_url": settings.OLLAMA_BASE_URL,
        "lm_studio_base_url": settings.LM_STUDIO_BASE_URL,
        "log_level": settings.LOG_LEVEL,
    }
