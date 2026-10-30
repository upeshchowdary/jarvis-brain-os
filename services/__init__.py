"""Background services: reminder scheduler, health monitor, clipboard history, workflow triggers, MCP servers."""

from __future__ import annotations

from core.events import wire_gateway
from core.logging import logger


async def start_background_services() -> None:
    from integrations.mcp_client import manager as mcp_manager
    from services.clipboard import clipboard_watcher
    from services.health import health_monitor
    from services.scheduler import scheduler
    from services.triggers import trigger_manager

    wire_gateway()
    await scheduler.start()
    await health_monitor.start()
    await clipboard_watcher.start()
    await trigger_manager.start()
    try:
        n = await mcp_manager.start()
        if n:
            logger.info(f"[Services] {n} MCP tools connected")
    except Exception as exc:
        logger.warning(f"[Services] MCP client failed: {exc}")
    logger.info("[Services] scheduler, health monitor, clipboard history and triggers running")


async def stop_background_services() -> None:
    from integrations.mcp_client import manager as mcp_manager
    from services.clipboard import clipboard_watcher
    from services.health import health_monitor
    from services.scheduler import scheduler
    from services.triggers import trigger_manager

    for svc in (scheduler, health_monitor, clipboard_watcher, trigger_manager):
        await svc.stop()
    await mcp_manager.stop()
