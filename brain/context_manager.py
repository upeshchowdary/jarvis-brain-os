"""Context Manager for assembling date/time, model, active task, history, and extensibility hooks."""

from typing import Dict, Any, List, Optional
from brain.utils import get_current_datetime_utc
from brain.model_manager import model_manager
from brain.logger import logger


class ContextManager:
    """Gathers date/time, model, task context, conversation history, and future screen/vision hooks."""

    def __init__(self) -> None:
        self._active_task: Optional[str] = None
        self._extensibility_hooks: Dict[str, Any] = {}

    def set_active_task(self, task_description: Optional[str]) -> None:
        """Set or update current active task metadata."""
        self._active_task = task_description

    def register_extensibility_hook(self, key: str, provider_callable: Any) -> None:
        """Register dynamic context hooks (for future active apps, screen OCR, memory)."""
        self._extensibility_hooks[key] = provider_callable

    def build_context(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Assemble comprehensive, extensible context dictionary."""
        from brain.utils import get_current_datetime_local
        current_dt = get_current_datetime_utc()
        local_dt = get_current_datetime_local()

        context = {
            "user_query": user_query,
            "current_datetime": current_dt,
            "current_date": local_dt["date"],
            "current_time": local_dt["time_12h_short"],
            "current_time_full": local_dt["time_12h"],
            "current_time_24h": local_dt["time_24h"],
            "current_full_date": local_dt["full_date"],
            "current_formatted_local": local_dt["formatted_full"],
            "active_model": model_manager.current_model,
            "active_task": self._active_task or "General Assistance",
            "conversation_history": conversation_history or [],
        }

        # Evaluate extensible hooks (e.g. active apps, screen analysis)
        for key, hook in self._extensibility_hooks.items():
            try:
                context[key] = hook() if callable(hook) else hook
            except Exception as exc:
                logger.warning(f"Error evaluating context hook '{key}': {exc}")
                context[key] = None

        if extra_context:
            context.update(extra_context)

        return context


# Global context manager singleton
context_manager = ContextManager()
