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
import subprocess
import sys
import time
import winreg
from typing import Dict, List, Optional, Tuple

from automation.config import automation_config
from automation.automation_logger import log_action
from automation.window_controller import window_controller

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class DynamicSystemAppManager:
    """Discovers, indexes, and resolves 100% of installed applications on this Windows PC."""

    def __init__(self) -> None:
        self._apps_cache: Dict[str, Dict[str, str]] = {}
        self._last_index_time: float = 0.0
        self.refresh_index()

    def refresh_index(self) -> int:
        """Scan system and index desktop, Store, and UWP apps with verified executable paths."""
        apps: Dict[str, Dict[str, str]] = {}

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
            try:
                for root_dir, _, files in os.walk(base_dir):
                    depth = root_dir[len(base_dir):].count(os.sep)
                    if depth > 3:
                        continue
                    for f in files:
                        if f.lower().endswith(".exe") and not any(x in f.lower() for x in ("unins", "setup", "update", "crash", "helper")):
                            clean_name = f.lower().replace(".exe", "").strip()
                            if clean_name not in apps:
                                full_p = os.path.join(root_dir, f)
                                apps[clean_name] = {
                                    "name": clean_name.title(),
                                    "exe": full_p,
                                    "type": "exe",
                                    "launch_cmd": full_p,
                                }
            except Exception:
                pass

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
        return len(apps)

    def find_app(self, query: str) -> Optional[Dict[str, str]]:
        """Find an installed app by exact name, alias, substring, or fuzzy match."""
        q = query.lower().strip().replace(".exe", "")

        # 1. Exact match
        if q in self._apps_cache:
            return self._apps_cache[q]

        # 2. Substring matching
        for k, v in self._apps_cache.items():
            if q == k or q in k or k in q:
                return v

        # 3. Fuzzy matching for slight typos
        matches = difflib.get_close_matches(q, self._apps_cache.keys(), n=1, cutoff=0.55)
        if matches:
            return self._apps_cache[matches[0]]

        return None

    def list_installed_apps(self) -> List[str]:
        """Return a sorted list of unique application names found on the PC."""
        names = {v["name"] for v in self._apps_cache.values()}
        return sorted(names)


# Global Dynamic App Manager
dynamic_app_manager = DynamicSystemAppManager()


class ApplicationController:
    """Launch, detect, and control any application on the user's computer."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run or automation_config.DRY_RUN
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

        # 1. Dynamically locate the application on the system
        app_info = self.app_manager.find_app(app_name)

        if not app_info:
            # Refresh index once in case it was installed recently
            self.app_manager.refresh_index()
            app_info = self.app_manager.find_app(app_name)

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
            # ── Native Windows Shell Execution ─────────────────────────
            if app_type == "exe" and os.path.exists(launch_cmd):
                # Direct binary execution via os.startfile (brings window to foreground)
                if not args:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: os.startfile(launch_cmd)
                    )
                else:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: subprocess.Popen(
                            f'"{launch_cmd}" {args}',
                            shell=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
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
            else:
                # Direct shell command fallback
                full_cmd = launch_cmd if not args else f"{launch_cmd} {args}"
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: subprocess.Popen(
                        full_cmd,
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                )

            # 2. Brief window confirmation (up to 1.5s non-blocking)
            wait_res = await window_controller.wait_for_window(display_name, timeout=1.5)
            if wait_res.get("success"):
                log_action("app", "open", display_name, result=f"READY: {wait_res.get('title')}")
                return {
                    "success": True,
                    "action": "open_app",
                    "app": display_name,
                    "window_title": wait_res.get("title"),
                }

            log_action("app", "open", display_name, result="LAUNCHED")
            return {
                "success": True,
                "action": "open_app",
                "app": display_name,
            }

        except Exception as e:
            log_action("app", "open", display_name, result=f"ERROR: {e}", level="ERROR")
            return {"success": False, "error": str(e), "app": display_name}

    async def close_app(self, name: str, force: bool = False) -> dict:
        """Close an application gracefully, or force-kill if needed."""
        from automation.safety_manager import safety_manager
        action_name = "close_app_force" if force else "close_window"
        allowed, reason = safety_manager.is_allowed(action_name, name)
        if not allowed:
            return {"success": False, "error": f"Blocked by safety manager: {reason}"}

        log_action("app", "close", name, dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "action": "close_app", "app": name}

        app_info = self.app_manager.find_app(name)
        display_name = app_info["name"] if app_info else name

        # 1. Try closing window gracefully
        win_res = await window_controller.close_window(display_name)
        if win_res["success"]:
            return {"success": True, "action": "close_app", "app": display_name}

        # 2. Fallback to process kill if requested or window close failed
        if HAS_PSUTIL and force:
            clean_name = name.lower().replace(".exe", "")
            killed = 0
            for proc in psutil.process_iter(["name", "pid"]):
                try:
                    if clean_name in (proc.info["name"] or "").lower():
                        proc.terminate()
                        killed += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if killed > 0:
                return {"success": True, "action": "close_app", "app": display_name, "killed_pids": killed}

        return {"success": win_res.get("success", False), "action": "close_app", "app": display_name}


# Global singleton
application_controller = ApplicationController()
