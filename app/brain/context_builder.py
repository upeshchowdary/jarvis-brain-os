"""Context Building Module for JARVIS Brain."""

from typing import Dict, Any, List
from app.domain.models.chat import ChatMessage, ChatRequest
from app.config.settings import settings
from app.tools.registry import tool_registry


class ContextBuilder:
    """Gathers application runtime state, user query context, and system metadata."""

    def build_context(self, request: ChatRequest) -> Dict[str, Any]:
        """Assemble full execution context dictionary for prompt compilation."""
        available_tools = tool_registry.list_tools()

        context = {
            "user_query": request.query,
            "session_id": request.session_id,
            "conversation_history": [m.model_dump() for m in request.conversation_history],
            "available_tools": available_tools,
            "system_config": {
                "app_name": settings.APP_NAME,
                "app_env": settings.APP_ENV,
                "provider": request.override_provider or settings.LLM_PROVIDER.value,
                "model": request.override_model or settings.MODEL_NAME,
            },
        }
        return context
