"""Health check (the only unauthenticated endpoint)."""

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter

from core.config import settings
from core.llm import resolve_model

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    provider, model = resolve_model(settings.DEFAULT_MODEL)
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.SYSTEM_VERSION,
        "active_provider": provider,
        "active_model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
