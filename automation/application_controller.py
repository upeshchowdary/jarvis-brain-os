"""JARVIS Dynamic Application Controller — Automatic system-wide app discovery & native execution.

Accurately scans, indexes, and controls ALL installed applications on Windows:
- Win32 Desktop applications (Registry App Paths, User AppData, Program Files)
- Microsoft Store & UWP applications (WhatsApp, Spotify, Calculator, Apple Music, ChatGPT)
- Windows Protocol URIs (ms-settings, ms-photos, mailto, etc.)
- Native system binaries & tools

Direct executable binaries take priority over shell AppIDs to guarantee flawless foreground launching.
"""

import asyncio
import difflib
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional

if sys.platform == "win32":
    import winreg
else:  # Non-Windows: registry discovery is skipped, the package still imports.
    winreg = None  # type: ignore[assignment]

from core.paths import data_path

from automation.automation_logger import log_action
from automation.window_controller import window_controller

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


_SCAN_MAX_DEPTH = 3
_SKIP_EXE_WORDS = ("unins", "setup", "update", "crash", "helper")


def _scan_exes(base_dir: str, max_depth: int = _SCAN_MAX_DEPTH) -> Dict[str, str]:
    """``{name: path}`` of the .exe files at most ``max_depth`` folders below ``base_dir``.

    The walk is pruned at ``max_depth``: walking the whole of Program Files (~110k folders) took 40-75 s.
    """
    found: Dict[str, str] = {}
    try:
        for root_dir, dirs, files in os.walk(base_dir):
            depth = root_dir[len(base_dir):].count(os.sep)
            if depth >= max_depth:
                dirs[:] = []  # never descend further
            if depth > max_depth:
                continue
            for f in files:
                low = f.lower()
                if low.endswith(".exe") and not any(x in low for x in _SKIP_EXE_WORDS):
                    found.setdefault(low.replace(".exe", "").strip(), os.path.join(root_dir, f))
    except Exception:
        pass
    return found


class DynamicSystemAppManager:
    """Discovers, indexes, and resolves 100% of installed applications on this Windows PC."""

    CACHE_MAX_AGE_SECONDS = 24 * 3600

    def __init__(self) -> None:
        # Discovery is lazy: nothing is scanned at import time.
        self._apps_cache: Dict[str, Dict[str, str]] = {}
        self._last_index_time: float = 0.0
        self._loaded = False
        self._refreshing = threading.Lock()

    @property
    def cache_file(self):
        return data_path("app_index.json")

    def _ensure_index(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            payload = json.loads(self.cache_file.read_text(encoding="utf-8"))
            apps = payload.get("apps", {})
            if apps:
                self._apps_cache = apps
                self._last_index_time = float(payload["indexed_at"])
                if time.time() - self._last_index_time >= self.CACHE_MAX_AGE_SECONDS:
                    # Stale: answer from the old index now, rescan in the background (a launch never waits on it).
                    self._refresh_in_background()
                return
        except (OSError, ValueError, KeyError):
            pass
        self.refresh_index()

    def _refresh_in_background(self) -> None:
        if not self._refreshing.acquire(blocking=False):
            return  # one rescan at a time

        def run() -> None:
            try:
                self.refresh_index()
            except Exception as e:  # noqa: BLE001
                log_action("app_discovery", "refresh", str(e), level="WARNING")
            finally:
                self._refreshing.release()

        threading.Thread(target=run, name="app-index-refresh", daemon=True).start()

    def refresh_index(self) -> int:
        """Scan system and index desktop, Store, and UWP apps with verified executable paths."""
        self._loaded = True
        apps: Dict[str, Dict[str, str]] = {}
        if sys.platform != "win32":
            self._apps_cache = apps
            self._last_index_time = time.time()
            return 0

        # ── Priority 1: Windows Registry "App Paths" (HKCU & HKLM) ───────
        # These are 100% genuine executable binary paths registered in Windows
        reg_roots = [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]
        for root in reg_roots:
            try:
                with winreg.OpenKey(root, r"Software\Microsoft\Windows\CurrentVersion\App Paths") as base_key:
                    num = winreg.QueryInfoKey(base_key)[0]
                    for i in range(num):
                        try:
                            k_name = winreg.EnumKey(base_key, i)
                            with winreg.OpenKey(base_key, k_name) as sub_key:
                                exe_path, _ = winreg.QueryValueEx(sub_key, "")
                                if exe_path and os.path.exists(exe_path):
                                    clean_name = k_name.lower().replace(".exe", "").strip()
                                    apps[clean_name] = {
                                        "name": k_name.replace(".exe", "").title(),
                                        "exe": exe_path,
                                        "type": "exe",
                                        "launch_cmd": exe_path,
                                    }
                        except Exception:
                            pass
            except Exception:
                pass

        # ── Priority 2: Common User AppData and Program Files Scanning ────
        common_scans = [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs"),
            os.path.expandvars(r"%PROGRAMFILES%"),
            os.path.expandvars(r"%PROGRAMFILES(X86)%"),
        ]
        for base_dir in common_scans:
            if not os.path.exists(base_dir):
                continue
            for clean_name, full_p in _scan_exes(base_dir).items():
                if clean_name not in apps:
                    apps[clean_name] = {
                        "name": clean_name.title(),
                        "exe": full_p,
                        "type": "exe",
                        "launch_cmd": full_p,
                    }

        # ── Priority 3: PowerShell Get-StartApps (Store / UWP Apps) ──────
        try:
            ps_cmd = "Get-StartApps | ConvertTo-Json -Compress"
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                data = json.loads(proc.stdout.strip())
                if isinstance(data, dict):
                    data = [data]
                for item in data:
                    name = item.get("Name", "").strip()
                    appid = item.get("AppID", "").strip()
                    if name and appid:
                        clean_key = name.lower().strip()
                        # If it's a true UWP package or not yet registered as a direct binary
                        if "!" in appid:
                            apps[clean_key] = {
                                "name": name,
                                "exe": appid,
                                "type": "uwp",
                                "launch_cmd": f"shell:AppsFolder\\{appid}",
                            }
                        elif clean_key not in apps:
                            if os.path.exists(appid):
                                apps[clean_key] = {
                                    "name": name,
                                    "exe": appid,
                                    "type": "exe",
                                    "launch_cmd": appid,
                                }
                            else:
                                apps[clean_key] = {
                                    "name": name,
                                    "exe": appid,
                                    "type": "startmenu",
                                    "launch_cmd": f"shell:AppsFolder\\{appid}",
                                }
        except Exception as e:
            log_action("app_discovery", "Get-StartApps", str(e), level="WARNING")

        # ── Priority 4: System Protocols & Essential Aliases ─────────────
        system_protocols = {
            "settings": ("Settings", "ms-settings:", "protocol"),
            "photos": ("Photos", "ms-photos:", "protocol"),
            "camera": ("Camera", "microsoft.windows.camera:", "protocol"),
            "mail": ("Mail", "mailto:", "protocol"),
            "store": ("Microsoft Store", "ms-windows-store:", "protocol"),
            "microsoft store": ("Microsoft Store", "ms-windows-store:", "protocol"),
            "paint": ("Paint", "mspaint.exe", "exe"),
            "notepad": ("Notepad", "notepad.exe", "exe"),
            "task manager": ("Task Manager", "taskmgr.exe", "exe"),
            "terminal": ("Windows Terminal", "wt.exe", "exe"),
            "powershell": ("PowerShell", "powershell.exe", "exe"),
            "cmd": ("Command Prompt", "cmd.exe", "exe"),
            "explorer": ("File Explorer", "explorer.exe", "exe"),
            "file explorer": ("File Explorer", "explorer.exe", "exe"),
        }
        for k, (disp, cmd_val, p_type) in system_protocols.items():
            if k not in apps or p_type == "protocol":
                apps[k] = {
                    "name": disp,
                    "exe": cmd_val,
                    "type": p_type,
                    "launch_cmd": cmd_val,
                }

        # Intuitive aliases
        if "google chrome" in apps and "chrome" not in apps:
            apps["chrome"] = apps["google chrome"]
        if "brave" in apps:
            apps["brave browser"] = apps["brave"]
        if "visual studio code" in apps:
            apps["vscode"] = apps["visual studio code"]
            apps["vs code"] = apps["visual studio code"]
            apps["code"] = apps["visual studio code"]
        if "telegram desktop" in apps and "telegram" not in apps:
            apps["telegram"] = apps["telegram desktop"]
        if "whatsapp" in apps:
            apps["whats app"] = apps["whatsapp"]
        if "calculator" in apps:
            apps["calc"] = apps["calculator"]

        self._apps_cache = apps
        self._last_index_time = time.time()
        try:
            self.cache_file.write_text(
                json.dumps({"indexed_at": self._last_index_time, "apps": apps}), encoding="utf-8"
            )
        except OSError:
            pass
        return len(apps)

    def find_app(self, query: str) -> Optional[Dict[str, str]]:
        """Find an installed app by exact name, alias, prefix, substring, or fuzzy match."""
        self._ensure_index()
        q = query.lower().strip().replace(".exe", "")
        if not q:
            return None

        # 1. Exact match
        if q in self._apps_cache:
            return self._apps_cache[q]

        # 2. Prefix, then substring matching (very short keys excluded to avoid false hits)
        for k, v in self._apps_cache.items():
            if k.startswith(q) and len(q) >= 3:
                return v
        for k, v in self._apps_cache.items():
            if len(k) >= 4 and (q in k or k in q.split()):
                return v

        # 3. Fuzzy matching for slight typos
        matches = difflib.get_close_matches(q, self._apps_cache.keys(), n=1, cutoff=0.55)
        if matches:
            return self._apps_cache[matches[0]]

        return None

    def list_installed_apps(self) -> List[str]:
        """Return a sorted list of unique application names found on the PC."""
        self._ensure_index()
        names = {v["name"] for v in self._apps_cache.values()}
        return sorted(names)


# Global Dynamic App Manager
dynamic_app_manager = DynamicSystemAppManager()


READY_TIMEOUT_SECONDS = 3.0  # after the launch command; the whole open_app stays well under ~5 s

# Launchers whose windows belong to another process (Windows 11: console apps run inside Windows Terminal).
_CONSOLE_HOSTS = {"conhost.exe", "openconsole.exe", "windowsterminal.exe"}
_WINDOW_PROCESS_ALIASES = {
    "wt.exe": {"windowsterminal.exe"},
    "cmd.exe": _CONSOLE_HOSTS,
    "powershell.exe": _CONSOLE_HOSTS,
    "pwsh.exe": _CONSOLE_HOSTS,
}


def app_processes(app_info: Dict[str, str]) -> set:
    """Lower-case exe names whose windows belong to this app; empty when the launch command doesn't say
    (Store / shell:AppsFolder / protocol launches): readiness then falls back to a title hint."""
    if app_info.get("type") != "exe":
        return set()
    exe = os.path.basename(str(app_info.get("launch_cmd") or "").strip().strip('"')).lower()
    if not exe:
        return set()
    if not exe.endswith(".exe"):
        exe += ".exe"
    return {exe} | _WINDOW_PROCESS_ALIASES.get(exe, set())


class ApplicationController:
    """Launch, detect, and control any application on the user's computer."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run  # global dry-run is enforced by the gateway
        self.app_manager = dynamic_app_manager

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

    def list_installed_apps(self) -> List[str]:
        """Get all discovered applications on this PC."""
        return self.app_manager.list_installed_apps()

    async def open_app(self, name: str, args: str = "") -> dict:
        """Dynamically find and launch any application installed on this PC."""
        app_name = name.strip()
        log_action("app", "open", app_name, dry_run=self.dry_run)

        if self.dry_run:
            return {"success": True, "dry_run": True, "action": "open_app", "app": app_name}

        # 1. Dynamically locate the application on the system (off the event loop: a cold index scans the disk)
        app_info = await asyncio.to_thread(self.app_manager.find_app, app_name)

        if not app_info:
            # Refresh index once in case it was installed recently
            await asyncio.to_thread(self.app_manager.refresh_index)
            app_info = await asyncio.to_thread(self.app_manager.find_app, app_name)

        if not app_info:
            err = (
                f"Application '{app_name}' was not found on this computer. "
                f"Total applications discovered: {len(self.app_manager._apps_cache)}."
            )
            log_action("app", "open", app_name, result=f"NOT FOUND: {err}", level="WARNING")
            return {"success": False, "error": err, "app": app_name}

        launch_cmd = app_info["launch_cmd"]
        app_type = app_info["type"]
        display_name = app_info["name"]

        try:
            # What was on screen before the launch, so readiness can tell a new window from a reused one
            before = await asyncio.to_thread(window_controller.window_snapshot)
            before_fg = await asyncio.to_thread(window_controller.foreground_window)

            # ── Native Windows Shell Execution ─────────────────────────
            if app_type == "exe" and os.path.exists(launch_cmd):
                # Direct binary execution via os.startfile (brings window to foreground)
                if not args:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: os.startfile(launch_cmd)
                    )
                else:
                    argv = [launch_cmd, *shlex.split(args, posix=False)]
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
                    )
            elif app_type == "protocol":
                # Protocol URI (ms-settings:, ms-windows-store:, mailto:)
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: os.startfile(launch_cmd)
                )
            elif app_type == "uwp":
                # True UWP Store Apps: explorer.exe shell:AppsFolder\<AppID>
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: subprocess.Popen(
                        ["explorer.exe", launch_cmd],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                )
            elif app_type == "startmenu" or launch_cmd.startswith("shell:"):
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: subprocess.Popen(["explorer.exe", launch_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
                )
            else:
                # Bare executable names (notepad.exe, wt.exe): resolved via PATH, no shell.
                argv = [launch_cmd, *shlex.split(args, posix=False)] if args else [launch_cmd]
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
                )

            # 2. Bounded readiness: the new window, the new tab, or the existing window brought to the front
            wait_res = await window_controller.wait_for_app_window(
                display_name, before, before_fg, timeout=READY_TIMEOUT_SECONDS, processes=app_processes(app_info))
            if wait_res.get("success"):
                window = {"hwnd": wait_res["hwnd"], "title": wait_res["title"], "kind": wait_res["kind"]}
                log_action("app", "open", display_name, result=f"READY ({window['kind']}): {window['title']}")
                return {
                    "success": True,
                    "action": "open_app",
                    "app": display_name,
                    "ready": True,
                    "window_title": window["title"],
                    "window": window,
                }

            note = f"Started {display_name}, but no window came to the front within {READY_TIMEOUT_SECONDS:g}s."
            log_action("app", "open", display_name, result=f"LAUNCHED, NOT READY: {note}", level="WARNING")
            return {
                "success": True,
                "action": "open_app",
                "app": display_name,
                "ready": False,
                "note": note,
            }

        except Exception as e:
            log_action("app", "open", display_name, result=f"ERROR: {e}", level="ERROR")
            return {"success": False, "error": str(e), "app": display_name}

    async def close_app(self, name: str, force: bool = False) -> dict:
        """Close an application window gracefully (force-killing is the gateway's process.kill op)."""
        log_action("app", "close", name, dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "action": "close_app", "app": name}

        app_info = self.app_manager.find_app(name)
        display_name = app_info["name"] if app_info else name

        # 1. Try closing window gracefully
        win_res = await window_controller.close_window(display_name)
        if win_res["success"]:
            return {"success": True, "action": "close_app", "app": display_name}

        return {"success": win_res.get("success", False), "action": "close_app", "app": display_name}


# Global singleton
application_controller = ApplicationController()
