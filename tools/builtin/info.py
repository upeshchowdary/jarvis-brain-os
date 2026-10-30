"""Read-only information tools: time/date, system status, windows, processes, apps."""

from __future__ import annotations

import asyncio
import platform
from datetime import datetime
from typing import Any, Dict

from tools.base import Tool, ToolContext, ToolResult, obj
from tools.registry import register


@register
class TimeDate(Tool):
    name = "get_time_date"
    description = "Get the current local date, time and weekday."
    parameters = obj({})

    async def run(self, ctx: ToolContext, **_: Any) -> ToolResult:
        now = datetime.now().astimezone()
        msg = f"It's {now.strftime('%I:%M %p').lstrip('0')} on {now.strftime('%A, %B %d, %Y')}."
        return ToolResult.ok(msg, data={"iso": now.isoformat()})


def collect_system_status() -> Dict[str, Any]:
    import psutil

    info: Dict[str, Any] = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "cpu_cores": psutil.cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "hostname": platform.node(),
    }
    mem = psutil.virtual_memory()
    info.update(ram_percent=mem.percent, ram_used_gb=round(mem.used / 1024**3, 1), ram_total_gb=round(mem.total / 1024**3, 1))
    try:
        disk = psutil.disk_usage("C:\\" if platform.system() == "Windows" else "/")
        info.update(disk_percent=disk.percent, disk_free_gb=round(disk.free / 1024**3, 1))
    except OSError:
        pass
    try:
        bat = psutil.sensors_battery()
    except Exception:
        bat = None
    if bat is not None:
        info.update(battery_percent=round(bat.percent), battery_plugged=bool(bat.power_plugged),
                    battery_minutes_left=(bat.secsleft // 60) if bat.secsleft and bat.secsleft > 0 else None)
    try:
        temps = psutil.sensors_temperatures() if hasattr(psutil, "sensors_temperatures") else {}
        hottest = max((t.current for entries in temps.values() for t in entries if t.current), default=None)
        if hottest:
            info["temperature_c"] = round(hottest)
    except Exception:
        pass
    return info


def format_status(info: Dict[str, Any], focus: str = "") -> str:
    focus = focus.lower()
    parts = []
    if "battery_percent" in info and focus in ("", "battery", "power"):
        plug = "charging" if info.get("battery_plugged") else "on battery"
        left = f", about {info['battery_minutes_left']} min left" if info.get("battery_minutes_left") else ""
        parts.append(f"Battery {info['battery_percent']}% ({plug}{left})")
    elif focus in ("battery", "power"):
        parts.append("No battery detected (desktop PC or unavailable).")
    if focus in ("", "cpu"):
        parts.append(f"CPU {info.get('cpu_percent')}% across {info.get('cpu_cores')} cores")
    if focus in ("", "ram", "memory"):
        parts.append(f"RAM {info.get('ram_used_gb')}/{info.get('ram_total_gb')} GB ({info.get('ram_percent')}%)")
    if focus in ("", "disk", "storage") and "disk_percent" in info:
        parts.append(f"Disk {info['disk_percent']}% used, {info.get('disk_free_gb')} GB free")
    if focus in ("", "temperature") and "temperature_c" in info:
        parts.append(f"Temperature {info['temperature_c']}°C")
    if focus in ("os", "system"):
        parts.append(f"{info.get('os')} on {info.get('hostname')}")
    return ". ".join(parts) + "." if parts else "System information is unavailable."


@register
class SystemStatus(Tool):
    name = "system_status"
    description = "Live CPU, RAM, disk, battery and temperature of this PC. focus: cpu|ram|disk|battery|temperature|os (optional)."
    parameters = obj({"focus": {"type": "string", "description": "Optional single metric to report"}})

    async def run(self, ctx: ToolContext, focus: str = "", **_: Any) -> ToolResult:
        info = await asyncio.to_thread(collect_system_status)
        return ToolResult.ok(format_status(info, focus), data=info)


@register
class ListWindows(Tool):
    name = "list_windows"
    description = "List the titles of open windows."
    parameters = obj({})

    async def run(self, ctx: ToolContext, **_: Any) -> ToolResult:
        from automation.window_controller import window_controller

        wins = await asyncio.to_thread(window_controller.list_windows)
        titles = [w.get("title", "")[:70] for w in wins if w.get("title")][:25]
        return ToolResult.ok("Open windows:\n" + "\n".join(f"• {t}" for t in titles) if titles else "I don't see any open windows.", data=titles)


@register
class ListProcesses(Tool):
    name = "list_processes"
    description = "List the processes using the most CPU and memory."
    parameters = obj({"limit": {"type": "integer", "default": 8}})

    async def run(self, ctx: ToolContext, limit: int = 8, **_: Any) -> ToolResult:
        import psutil

        def _top():
            procs = []
            for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                if p.info.get("name"):
                    procs.append(p.info)
            return sorted(procs, key=lambda p: ((p.get("memory_percent") or 0)), reverse=True)[: int(limit)]

        top = await asyncio.to_thread(_top)
        lines = [f"• {p['name']} (PID {p['pid']}) — RAM {round(p.get('memory_percent') or 0, 1)}%" for p in top]
        return ToolResult.ok("Top processes by memory:\n" + "\n".join(lines), data=top)


@register
class ListApps(Tool):
    name = "list_installed_apps"
    description = "List applications installed on this PC (optionally filtered by a search term)."
    parameters = obj({"filter": {"type": "string"}})

    async def run(self, ctx: ToolContext, filter: str = "", **_: Any) -> ToolResult:
        from automation.application_controller import application_controller

        apps = await asyncio.to_thread(application_controller.list_installed_apps)
        if filter:
            apps = [a for a in apps if filter.lower() in a.lower()]
        sample = ", ".join(apps[:40])
        return ToolResult.ok(f"Found {len(apps)} apps: {sample}{' …' if len(apps) > 40 else ''}" if apps else "No matching apps found.", data=apps)
