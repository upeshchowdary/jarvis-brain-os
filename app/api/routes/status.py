"""Telemetry and Status REST Endpoint."""

from fastapi import APIRouter, Depends
from typing import Dict, Any
from app.api.dependencies import get_database_manager
from app.database.connection import DatabaseManager
from app.tools.registry import tool_registry

router = APIRouter(tags=["Status"])


@router.get("/status", response_model=Dict[str, Any])
async def get_system_status(
    db: DatabaseManager = Depends(get_database_manager),
) -> Dict[str, Any]:
    """Return execution metrics, registered tool count, and recent chat history logs."""
    recent_logs = await db.get_recent_logs(limit=10)
    registered_tools = tool_registry.list_tools()

    return {
        "active_modules": [
            "BrainOrchestrator",
            "IntentDetector",
            "ContextBuilder",
            "ReasoningEngine",
            "Planner",
            "PromptManager",
            "LLMFactory",
            "DatabaseManager",
        ],
        "registered_tool_count": len(registered_tools),
        "registered_tools": registered_tools,
        "recent_execution_count": len(recent_logs),
        "recent_executions": recent_logs,
    }
