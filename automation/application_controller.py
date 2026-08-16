"""JARVIS Application Controller — Launch, detect, and manage applications.

Supports common Windows applications with a registry of known paths.
Does NOT assume fixed startup time — uses state verification instead.
"""

import asyncio
import subprocess
import sys
from typing import Optional, Dict

from automation.config import automation_config
from automation.automation_logger import log_action
from automation.window_controller import window_controller

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ── Known application registry ───────────────────────────────────────────────
# Maps common app names to executable paths / shell commands on Windows

_APP_REGISTRY: Dict[str, str] = {
    # Browsers
    "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "google chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "firefox": r"C:\Program Files\Mozilla Firefox\firefox.exe",
    "edge": r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "microsoft edge": r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "brave": r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",

    # Development
    "vscode": r"code",
    "vs code": r"code",
    "visual studio code": r"code",
    "notepad": r"notepad.exe",
    "notepad++": r"C:\Program Files\Notepad++\notepad++.exe",

    # System
    "explorer": r"explorer.exe",
    "file explorer": r"explorer.exe",
    "cmd": r"cmd.exe",
    "command prompt": r"cmd.exe",
    "powershell": r"powershell.exe",
    "terminal": r"wt.exe",  # Windows Terminal
    "windows terminal": r"wt.exe",
    "settings": r"ms-settings:",
    "task manager": r"taskmgr.exe",
    "calculator": r"calc.exe",
    "paint": r"mspaint.exe",
    "wordpad": r"wordpad.exe",

    # Office
    "word": r"winword.exe",
    "excel": r"excel.exe",
    "powerpoint": r"powerpnt.exe",
    "outlook": r"outlook.exe",

    # Communication
    "teams": r"ms-teams:",
    "discord": r"discord.exe",
    "slack": r"slack.exe",
    "zoom": r"zoom.exe",
    "whatsapp": r"WhatsApp.exe",
    "telegram": r"telegram.exe",

    # Media
    "vlc": r"C:\Program Files\VideoLAN\VLC\vlc.exe",
    "spotify": r"spotify.exe",
    "photos": r"ms-photos:",

    # Other
    "obs": r"obs64.exe",
    "paint 3d": r"ms-paint:",
    "snipping tool": r"snippingtool.exe",
}

# Window title fragments to detect each app after launch
_APP_WINDOW_HINTS: Dict[str, str] = {
    "chrome": "Google Chrome",
    "google chrome": "Google Chrome",
    "firefox": "Mozilla Firefox",
    "edge": "Edge",
    "microsoft edge": "Microsoft Edge",
    "brave": "Brave",
    "vscode": "Visual Studio Code",
    "vs code": "Visual Studio Code",
    "visual studio code": "Visual Studio Code",
    "notepad": "Notepad",
    "notepad++": "Notepad++",
    "explorer": "File Explorer",
    "file explorer": "File Explorer",
    "cmd": "Command Prompt",
    "command prompt": "Command Prompt",
    "powershell": "Windows PowerShell",
    "terminal": "Windows PowerShell",
    "calculator": "Calculator",
    "paint": "Paint",
    "word": "Word",
    "excel": "Excel",
    "teams": "Teams",
    "discord": "Discord",
    "slack": "Slack",
    "zoom": "Zoom",
}


class ApplicationController:
    """Launch, detect, and wait for application readiness."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run or automation_config.DRY_RUN

    def _resolve_app(self, name: str) -> Optional[str]:
        """Resolve an app name to its executable path."""
        return _APP_REGISTRY.get(name.lower().strip())

    def is_running(self, process_name: str) -> bool:
        """Check if a process is currently running by name."""
        if not HAS_PSUTIL:
            return False
        proc_lower = process_name.lower().replace(".exe", "")
        for proc in psutil.process_iter(["name"]):
            try:
                if proc_lower in (proc.info["name"] or "").lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return False

    async def open_app(self, name: str, args: str = "") -> dict:
        """Open an application by name. Waits for it to appear on screen."""
        app_key = name.lower().strip()
        executable = self._resolve_app(app_key)

        log_action("app", "open", name, dry_run=self.dry_run)

        if self.dry_run:
            return {"success": True, "dry_run": True, "action": "open_app", "app": name}

        if not executable:
            # Try running it directly — might be in PATH
            executable = name

        try:
            cmd = executable
            if args:
                cmd = f"{executable} {args}"

            # Use shell=True for ms- protocol URIs and apps in PATH
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )

            # Wait for the window to appear
            window_hint = _APP_WINDOW_HINTS.get(app_key, name)
            wait_result = await window_controller.wait_for_window(
                window_hint,
                timeout=automation_config.APP_STARTUP_TIMEOUT
            )

            if wait_result["success"]:
                log_action("app", "open", name, result=f"READY: {wait_result.get('title')}")
                return {"success": True, "action": "open_app", "app": name,
                        "window_title": wait_result.get("title")}
            else:
                # App may have launched but window title differs — return partial success
                log_action("app", "open", name, result="LAUNCHED (window not confirmed)")
                return {"success": True, "action": "open_app", "app": name,
                        "note": "App launched but window title not confirmed"}

        except Exception as e:
            log_action("app", "open", name, result=f"ERROR: {e}", level="ERROR")
            return {"success": False, "error": str(e), "app": name}

    async def close_app(self, name: str, force: bool = False) -> dict:
        """Close an application gracefully, or force-kill if needed."""
        from automation.safety_manager import safety_manager
        action_name = "close_app_force" if force else "close_window"
        allowed, reason = safety_manager.is_allowed(action_name, name)
        if not allowed:
            return {"success": False, "blocked": True, "reason": reason}

        log_action("app", "close" + (" (force)" if force else ""), name, dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}

        if not HAS_PSUTIL or not force:
            # Try graceful window close first
            result = await window_controller.close_window(name)
            return result

        # Force kill via psutil
        name_lower = name.lower().replace(".exe", "")
        killed = 0
        for proc in psutil.process_iter(["name", "pid"]):
            try:
                if name_lower in (proc.info["name"] or "").lower():
                    proc.kill()
                    killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return {"success": True, "action": "close_app", "processes_killed": killed}

    async def wait_for_app(self, name: str, timeout: float | None = None) -> bool:
        """Wait until an app's window appears. Returns True if found, False on timeout."""
        window_hint = _APP_WINDOW_HINTS.get(name.lower().strip(), name)
        result = await window_controller.wait_for_window(
            window_hint, timeout=timeout or automation_config.APP_STARTUP_TIMEOUT
        )
        return result["success"]

    def list_known_apps(self) -> list:
        """Return list of app names JARVIS knows how to open."""
        return sorted(set(_APP_REGISTRY.keys()))


# Global singleton
application_controller = ApplicationController()
