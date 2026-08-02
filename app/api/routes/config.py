"""System Configuration REST Endpoint."""

from fastapi import APIRouter
from typing import Dict, Any
from app.config.settings import settings

router = APIRouter(tags=["Configuration"])


@router.get("/config", response_model=Dict[str, Any])
async def get_configuration() -> Dict[str, Any]:
    """Return non-sensitive system settings and runtime configurations."""
    return {
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "debug_mode": settings.DEBUG,
        "default_llm_provider": settings.LLM_PROVIDER.value,
        "default_model": settings.MODEL_NAME,
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "timeout_seconds": settings.LLM_TIMEOUT_SECONDS,
        "ollama_base_url": settings.OLLAMA_BASE_URL,
        "lm_studio_base_url": settings.LM_STUDIO_BASE_URL,
        "log_level": settings.LOG_LEVEL,
    }
