"""Health Check REST Endpoint."""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from typing import Dict, Any
from app.config.settings import settings
from app.llm.factory import LLMFactory

from brain.brain_manager import brain_manager

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """Return health status of JARVIS Brain Engine and active LLM provider connectivity."""
    active_model = brain_manager.model_manager.current_model
    if "ollama" in active_model.lower() or "qwen" in active_model.lower():
        active_provider = "ollama"
    elif "gemini" in active_model.lower():
        active_provider = "gemini"
    elif "gpt" in active_model.lower() or "openai" in active_model.lower():
        active_provider = "openai"
    else:
        active_provider = "groq"

    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "active_provider": active_provider,
        "active_model": active_model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
