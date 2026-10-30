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
        "default_llm_provider": settings.DEFAULT_MODEL.split("/", 1)[0],
        "default_model": settings.DEFAULT_MODEL,
        "temperature": settings.TEMPERATURE,
        "max_tokens": settings.MAX_TOKENS,
        "timeout_seconds": settings.TIMEOUT_SECONDS,
        "ollama_base_url": settings.OLLAMA_BASE_URL,
        "lm_studio_base_url": settings.LM_STUDIO_BASE_URL,
        "log_level": settings.LOG_LEVEL,
    }
