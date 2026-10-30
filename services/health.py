"""Background system-health watcher: battery, CPU, RAM, disk, temperature → proactive alerts.

Alerts are throttled (one per kind every 15 min) and only suggest fixes; ending a
process still goes through the gateway with confirmation ("kill chrome.exe").
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional

from core.config import settings
from core.logging import logger
from services.notify import notify

THROTTLE_SECONDS = 15 * 60


def evaluate(info: Dict, top: Optional[List[Dict]] = None) -> List[tuple[str, str]]:
    """Pure check: returns (kind, message) alerts for a status snapshot."""
    alerts: List[tuple[str, str]] = []
    bat = info.get("battery_percent")
    if bat is not None and not info.get("battery_plugged") and bat <= settings.HEALTH_BATTERY_ALERT:
        alerts.append(("battery", f"Battery is at {bat}%. Plug in your charger."))
    hog = f" Top process: {top[0]['name']} — say 'kill {top[0]['name']}' to end it (I'll ask first)." if top else ""
    if info.get("cpu_percent", 0) >= settings.HEALTH_CPU_ALERT:
        alerts.append(("cpu", f"CPU has been at {info['cpu_percent']}%.{hog}"))
    if info.get("ram_percent", 0) >= settings.HEALTH_RAM_ALERT:
        alerts.append(("ram", f"Memory is {info['ram_percent']}% full.{hog}"))
    if info.get("disk_percent", 0) >= settings.HEALTH_DISK_ALERT:
        alerts.append(("disk", f"Your disk is {info['disk_percent']}% full ({info.get('disk_free_gb')} GB free). "
                               "Say 'find duplicates in downloads' or 'organize my downloads' to clean up."))
    if info.get("temperature_c", 0) >= 90:
        alerts.append(("temperature", f"Your PC is running hot ({info['temperature_c']}°C)."))
    return alerts


class HealthMonitor:
    def __init__(self, interval: float = 30.0) -> None:
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self._last: Dict[str, float] = {}
        self._cpu_strikes = 0

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        self._task = None

    async def check_once(self) -> List[tuple[str, str]]:
        import psutil

        from tools.builtin.info import collect_system_status

        info = await asyncio.to_thread(collect_system_status)
        # CPU must stay high for 3 consecutive checks before alerting
        self._cpu_strikes = self._cpu_strikes + 1 if info.get("cpu_percent", 0) >= settings.HEALTH_CPU_ALERT else 0
        if self._cpu_strikes < 3:
            info["cpu_percent"] = min(info.get("cpu_percent", 0), settings.HEALTH_CPU_ALERT - 0.1)

        def _top():
            procs = [p.info for p in psutil.process_iter(["name", "memory_percent"]) if p.info.get("name")]
            return sorted(procs, key=lambda p: p.get("memory_percent") or 0, reverse=True)[:1]

        alerts = evaluate(info, await asyncio.to_thread(_top))
        now = time.time()
        fresh = [(k, m) for k, m in alerts if now - self._last.get(k, 0) > THROTTLE_SECONDS]
        for kind, message in fresh:
            self._last[kind] = now
            await notify("⚠️ System health", message, kind="health_alert")
        return fresh

    async def _loop(self) -> None:
        while True:
            try:
                await self.check_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug(f"[Health] check failed: {exc}")
            await asyncio.sleep(self.interval)


health_monitor = HealthMonitor()
