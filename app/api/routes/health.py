"""Health Check REST Endpoint."""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from typing import Dict, Any
from app.config.settings import settings
from app.llm.factory import LLMFactory

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """Return health status of JARVIS Brain Engine and active LLM provider connectivity."""
    provider = LLMFactory.create_provider()
    provider_healthy = await provider.health_check()

    return {
        "status": "healthy" if provider_healthy else "degraded",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "active_provider": settings.LLM_PROVIDER.value,
        "provider_healthy": provider_healthy,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
