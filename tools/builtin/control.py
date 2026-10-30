"""Computer-control tools. Every one of them acts *only* through the ActionGateway."""

from __future__ import annotations

from typing import Any

from automation.gateway import ActionRequest, Risk, gateway
from core.config import settings
from tools.base import Tool, ToolContext, ToolResult, obj, via_gateway
from tools.registry import register


class GatewayTool(Tool):
    side_effect = True
    risk = Risk.LOW


@register
class Volume(GatewayTool):
    name = "set_volume"
    description = "Change system volume: action up|down|set|mute|unmute|get, with level 0-100 for set or step for up/down."
    parameters = obj({
        "action": {"type": "string", "enum": ["up", "down", "set", "mute", "unmute", "get"]},
        "level": {"type": "integer", "minimum": 0, "maximum": 100},
        "step": {"type": "integer", "minimum": 1, "maximum": 100},
    }, ["action"])

    async def run(self, ctx: ToolContext, action: str = "set", level: Any = None, step: Any = 10, **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("system.volume", {"action": action, "level": level, "step": step or 10}))


@register
class Media(GatewayTool):
    name = "media_control"
    description = "Media playback: play_pause | next | previous."
    parameters = obj({"action": {"type": "string", "enum": ["play_pause", "next", "previous"]}}, ["action"])

    async def run(self, ctx: ToolContext, action: str = "play_pause", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("system.media", {"action": action}))


@register
class Brightness(GatewayTool):
    name = "set_brightness"
    description = "Change screen brightness: action up|down|set|get, level 0-100 for set."
    parameters = obj({
        "action": {"type": "string", "enum": ["up", "down", "set", "get"]},
        "level": {"type": "integer", "minimum": 0, "maximum": 100},
        "step": {"type": "integer", "minimum": 1, "maximum": 100},
    }, ["action"])

    async def run(self, ctx: ToolContext, action: str = "set", level: Any = None, step: Any = 10, **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("system.brightness", {"action": action, "level": level, "step": step or 10}))


@register
class WindowControl(GatewayTool):
    name = "window_control"
    description = "Manage windows: minimize|maximize|restore|close|focus|snap|show_desktop|switch|list. target = window/app title (empty = active window)."
    parameters = obj({
        "action": {"type": "string", "enum": ["minimize", "maximize", "restore", "close", "focus", "snap", "show_desktop", "switch", "list"]},
        "target": {"type": "string"},
        "direction": {"type": "string", "enum": ["left", "right", "up", "down"]},
    }, ["action"])

    async def run(self, ctx: ToolContext, action: str = "", target: str = "", direction: str = "left", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("window.manage", {"action": action, "target": target, "direction": direction}))


@register
class LockScreen(GatewayTool):
    name = "lock_screen"
    description = "Lock the PC screen."
    parameters = obj({})

    async def run(self, ctx: ToolContext, **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("system.lock"))


@register
class Power(GatewayTool):
    name = "power_control"
    description = "Shut down, restart, sleep, hibernate or sign out of the PC. Always asks the user to confirm. Only use when explicitly asked."
    parameters = obj({"action": {"type": "string", "enum": ["shutdown", "restart", "sleep", "hibernate", "logoff"]}}, ["action"])
    risk = Risk.HIGH

    async def run(self, ctx: ToolContext, action: str = "", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("system.power", {"action": action}))


@register
class CancelShutdown(GatewayTool):
    name = "cancel_shutdown"
    description = "Cancel a scheduled shutdown or restart."
    parameters = obj({})

    async def run(self, ctx: ToolContext, **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("system.cancel_shutdown"))


@register
class OpenApp(GatewayTool):
    name = "open_app"
    description = "Launch an installed application by name (e.g. 'chrome', 'vs code', 'spotify', 'calculator')."
    parameters = obj({"name": {"type": "string"}}, ["name"])

    async def run(self, ctx: ToolContext, name: str = "", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("app.open", {"name": name}))


@register
class CloseApp(GatewayTool):
    name = "close_app"
    description = "Close an application window gracefully."
    parameters = obj({"name": {"type": "string"}}, ["name"])
    risk = Risk.MEDIUM

    async def run(self, ctx: ToolContext, name: str = "", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("app.close", {"name": name}))


@register
class OpenWebsite(GatewayTool):
    name = "open_website"
    description = "Open a website or URL in the browser (e.g. 'youtube', 'github.com', 'https://…')."
    parameters = obj({"url": {"type": "string"}, "browser": {"type": "string", "enum": ["", "chrome", "edge", "firefox", "brave"]}}, ["url"])

    async def run(self, ctx: ToolContext, url: str = "", browser: str = "", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("browser.open_url", {"url": url, "browser": browser}))


@register
class BrowserSearch(GatewayTool):
    name = "search_in_browser"
    description = "Open a search results page in the browser (google, youtube, maps, amazon, github, wikipedia)."
    parameters = obj({"query": {"type": "string"}, "engine": {"type": "string", "enum": ["google", "youtube", "maps", "amazon", "github", "wikipedia", "bing"]}}, ["query"])

    async def run(self, ctx: ToolContext, query: str = "", engine: str = "google", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("browser.search", {"query": query, "engine": engine or "google"}))


@register
class KillProcess(GatewayTool):
    name = "kill_process"
    description = "Force-end a running program by its exact executable name (e.g. 'notepad.exe') or PID. Always confirmed."
    parameters = obj({"target": {"type": "string"}}, ["target"])
    risk = Risk.HIGH

    async def run(self, ctx: ToolContext, target: str = "", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("process.kill", {"target": target}))


@register
class Screenshot(GatewayTool):
    name = "take_screenshot"
    description = "Save a screenshot of the screen to disk."
    parameters = obj({})

    async def run(self, ctx: ToolContext, **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("system.screenshot"))


@register
class Network(GatewayTool):
    name = "network_toggle"
    description = "Turn Wi-Fi or Bluetooth on/off. Turning off asks for confirmation."
    parameters = obj({"device": {"type": "string", "enum": ["wifi", "bluetooth"]}, "enable": {"type": "boolean"}}, ["device", "enable"])

    async def run(self, ctx: ToolContext, device: str = "wifi", enable: bool = True, **_: Any) -> ToolResult:
        op = "network.bluetooth" if device == "bluetooth" else "network.wifi"
        return await via_gateway(ctx, ActionRequest(op, {"enable": bool(enable)}))


@register
class OpenSystemTool(GatewayTool):
    name = "open_system_tool"
    description = "Open Task Manager, Windows Settings (optional page like 'display', 'bluetooth', 'network') or File Explorer (optional path)."
    parameters = obj({"tool": {"type": "string", "enum": ["task_manager", "settings", "file_explorer"]}, "page": {"type": "string"}}, ["tool"])

    async def run(self, ctx: ToolContext, tool: str = "", page: str = "", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("system.open_tool", {"tool": tool, "page": page}))


@register
class EmptyRecycleBin(GatewayTool):
    name = "empty_recycle_bin"
    description = "Permanently empty the Recycle Bin. Always confirmed."
    parameters = obj({})
    risk = Risk.HIGH

    async def run(self, ctx: ToolContext, **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("system.empty_recycle_bin"))


@register
class RunNamedCommand(GatewayTool):
    name = "run_named_command"
    description = (
        "Run one allow-listed command: ipconfig, ping(host), flush_dns, git_status(path), git_log_today(path), "
        "git_diff_stat(path), pytest(path), npm_test(path). Arbitrary shell commands are not possible."
    )
    parameters = obj({"name": {"type": "string"}, "params": {"type": "object", "additionalProperties": {"type": "string"}}}, ["name"])
    risk = Risk.HIGH

    async def run(self, ctx: ToolContext, name: str = "", params: Any = None, **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("shell.run_named", {"name": name, "params": dict(params or {})}))


@register
class ComputerTask(GatewayTool):
    name = "computer_task"
    description = (
        "Perform a multi-step task in apps or websites by clicking and typing (e.g. 'open chrome and search for X then "
        "open the first result'). The plan is shown to the user for confirmation before it runs."
    )
    parameters = obj({"goal": {"type": "string", "description": "The full task in plain language"}}, ["goal"])
    risk = Risk.MEDIUM

    async def run(self, ctx: ToolContext, goal: str = "", **_: Any) -> ToolResult:
        from automation.engine.engine_orchestrator import advanced_orchestrator
        from tools.base import from_gateway

        return from_gateway(await advanced_orchestrator.execute(goal, ctx.session_id, dry_run=ctx.dry_run, source=ctx.source))


@register
class AutomationControl(Tool):
    """Emergency stop / resume / dry-run — controls the safety system itself (never needs confirmation).

    Marked side_effect so content from web pages/files (taint) can never toggle it.
    """

    name = "automation_control"
    description = "Control automation safety: stop (emergency stop all actions), resume, dry_run_on, dry_run_off."
    parameters = obj({"action": {"type": "string", "enum": ["stop", "resume", "dry_run_on", "dry_run_off"]}}, ["action"])
    side_effect = True

    async def run(self, ctx: ToolContext, action: str = "status", **_: Any) -> ToolResult:
        if action == "stop":
            gateway.emergency_stop(f"user ({ctx.channel})")
            return ToolResult.ok("🛑 Emergency stop: all automation halted and pending actions cancelled. Say 'resume automation' to continue.")
        if action == "resume":
            gateway.resume()
            return ToolResult.ok("Automation resumed.")
        if action in ("dry_run_on", "dry_run_off"):
            settings.DRY_RUN = action == "dry_run_on"
            return ToolResult.ok(f"Dry-run mode {'ON — actions will only be simulated' if settings.DRY_RUN else 'OFF — actions will run for real'}.")
        return await AutomationStatus().run(ctx)


@register
class AutomationStatus(Tool):
    name = "automation_status"
    description = "Report whether emergency stop / dry-run are active and whether an action is awaiting confirmation."
    parameters = obj({})

    async def run(self, ctx: ToolContext, **_: Any) -> ToolResult:
        pending = gateway.pending_for(ctx.session_id)
        return ToolResult.ok(
            f"Emergency stop: {'ACTIVE' if gateway.is_stopped else 'clear'}. Dry-run: {'on' if settings.DRY_RUN else 'off'}. "
            f"Medium-risk confirmation: {'on' if settings.CONFIRM_MEDIUM_RISK else 'off'}. "
            f"Pending confirmation: {pending.preview if pending else 'none'}."
        )
