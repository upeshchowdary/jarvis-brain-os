"""Background services (scheduler, health monitor, clipboard history, triggers)."""

from __future__ import annotations

from typing import List

from core.logging import logger

_started: List[object] = []


async def start_background_services() -> None:
    logger.info("[Services] background services: none registered yet")


async def stop_background_services() -> None:
    _started.clear()
