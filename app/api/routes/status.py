"""Runtime status: tools, recent executions, sessions, metrics."""

from typing import Any, Dict

from fastapi import APIRouter, Depends

from app.api.dependencies import get_database_manager
from app.database.connection import DatabaseManager
from core.metrics import metrics
from core.session import sessions
from tools.registry import load_builtin_tools

router = APIRouter(tags=["Status"])


@router.get("/status")
async def get_system_status(db: DatabaseManager = Depends(get_database_manager)) -> Dict[str, Any]:
    registry = load_builtin_tools()
    recent_logs = await db.get_recent_logs(limit=10)
    return {
        "active_modules": ["Router", "AgentPipeline", "LLMGateway", "ActionGateway", "Memory", "Vision", "Voice", "Scheduler"],
        "registered_tool_count": len(registry.all()),
        "registered_tools": registry.info(),
        "active_sessions": len(sessions),
        "recent_execution_count": len(recent_logs),
        "recent_executions": recent_logs,
    }


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Latency per stage (p50/p95), provider usage and error counters."""
    return metrics.snapshot()
