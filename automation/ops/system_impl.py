"""Low-level Windows system primitives (PRIVATE — use automation.gateway).

Only ``automation.ops.system`` may import this module; it wraps every function
below in a gateway operation with a risk level.

Provides programmatic access to every major system function:
- Window management (minimize, maximize, close, focus, resize, list, switch)
- Volume control (get, set, mute, unmute, up, down)
- Brightness control (get, set, up, down)
- Power management (shutdown, restart, sleep, hibernate, lock, logoff)
- Clipboard (get, set, clear)
- Screenshot & screen recording
- Process management (list, kill, start)
- Network (WiFi on/off, list networks, connect)
- System info (CPU, RAM, disk, battery, temperature)
- Display (resolution change, multiple monitors)
- Notifications
- Taskbar, Start Menu, virtual desktops
- Hotkeys & global shortcuts
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

from automation.automation_logger import automation_logger


# ─── Volume Control ──────────────────────────────────────────────────────────

def get_volume() -> int:
    """Return current system volume (0-100)."""
    try:
        import comtypes  # type: ignore
        comtypes.CoInitialize()
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL  # type: ignore
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # type: ignore
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        val = int(volume.GetMasterVolumeLevelScalar() * 100)
        del volume, interface, devices
        return val
    except Exception:
        return 50


def set_volume(level: int) -> bool:
    """Set system volume to level (0-100)."""
    level = max(0, min(100, level))
    try:
        import comtypes  # type: ignore
        comtypes.CoInitialize()
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL  # type: ignore
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # type: ignore
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume.SetMasterVolumeLevelScalar(level / 100.0, None)
        del volume, interface, devices
        return True
    except Exception:
        return False




def set_volume_via_keys(direction: str, steps: int = 5) -> bool:
    """Adjust volume using media keys (up/down/mute)."""
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        import win32api
        import win32con
        VK_VOLUME_UP   = 0xAF
        VK_VOLUME_DOWN = 0xAE
        VK_VOLUME_MUTE = 0xAD
        if direction == "up":
            for _ in range(steps):
                win32api.keybd_event(VK_VOLUME_UP, 0, 0, 0)
                win32api.keybd_event(VK_VOLUME_UP, 0, win32con.KEYEVENTF_KEYUP, 0)
        elif direction == "down":
            for _ in range(steps):
                win32api.keybd_event(VK_VOLUME_DOWN, 0, 0, 0)
                win32api.keybd_event(VK_VOLUME_DOWN, 0, win32con.KEYEVENTF_KEYUP, 0)
        elif direction == "mute":
            win32api.keybd_event(VK_VOLUME_MUTE, 0, 0, 0)
            win32api.keybd_event(VK_VOLUME_MUTE, 0, win32con.KEYEVENTF_KEYUP, 0)
        return True
    except Exception as e:
        automation_logger.warning(f"[SystemController] Volume key error: {e}")
        return False


# ─── Brightness Control ───────────────────────────────────────────────────────

def get_brightness() -> int:
    """Get current screen brightness (0-100)."""
    try:
        import wmi  # type: ignore
        c = wmi.WMI(namespace="wmi")
        methods = c.WmiMonitorBrightness()
        if methods:
            return methods[0].CurrentBrightness
    except Exception:
        pass
    return -1


def set_brightness(level: int) -> bool:
    """Set screen brightness to level (0-100)."""
    level = max(0, min(100, level))
    try:
        import wmi  # type: ignore
        c = wmi.WMI(namespace="wmi")
        methods = c.WmiMonitorBrightnessMethods()
        if methods:
            methods[0].WmiSetBrightness(level, 0)
            return True
    except Exception:
        pass
    # Fallback: PowerShell
    try:
        subprocess.run(
            ["powershell", "-Command", f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1, {level})"],
            capture_output=True, timeout=5
        )
        return True
    except Exception:
        return False


def adjust_brightness(direction: str, steps: int = 10) -> bool:
    """Increase or decrease brightness by steps."""
    current = get_brightness()
    if current < 0:
        current = 50
    if direction == "up":
        return set_brightness(min(100, current + steps))
    elif direction == "down":
        return set_brightness(max(0, current - steps))
    return False


# ─── Window Management ────────────────────────────────────────────────────────

def get_all_windows() -> List[Dict[str, Any]]:
    """List all open windows with their titles and state."""
    try:
        import pygetwindow as gw  # type: ignore
        wins = gw.getAllWindows()
        return [
            {
                "title": w.title,
                "left": w.left, "top": w.top,
                "width": w.width, "height": w.height,
                "visible": w.visible, "active": w.isActive,
            }
            for w in wins if w.title.strip()
        ]
    except Exception:
        return []


def window_action(title_fragment: str, action: str) -> bool:
    """Perform minimize/maximize/close/restore on a window matching title_fragment."""
    try:
        import pygetwindow as gw  # type: ignore
        matches = gw.getWindowsWithTitle(title_fragment)
        if not matches:
            # Try to find any window containing fragment
            all_wins = gw.getAllWindows()
            matches = [w for w in all_wins if title_fragment.lower() in w.title.lower() and w.title.strip()]
        if not matches:
            return False
        w = matches[0]
        if action == "minimize":
            w.minimize()
        elif action == "maximize":
            w.maximize()
        elif action == "restore":
            w.restore()
        elif action == "close":
            w.close()
        elif action == "focus" or action == "activate":
            w.activate()
        return True
    except Exception as e:
        automation_logger.warning(f"[SystemController] Window action '{action}' error: {e}")
        return False


def minimize_active_window() -> bool:
    """Minimize the currently active/focused window."""
    try:
        import win32gui
        import win32con
        hwnd = win32gui.GetForegroundWindow()
        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
        return True
    except Exception:
        try:
            import pyautogui
            pyautogui.FAILSAFE = False
            pyautogui.hotkey("win", "down")
            return True
        except Exception:
            return False


def maximize_active_window() -> bool:
    """Maximize the currently active window."""
    try:
        import win32gui
        import win32con
        hwnd = win32gui.GetForegroundWindow()
        win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
        return True
    except Exception:
        try:
            import pyautogui
            pyautogui.FAILSAFE = False
            pyautogui.hotkey("win", "up")
            return True
        except Exception:
            return False


def close_active_window() -> bool:
    """Close the currently active window (Alt+F4)."""
    try:
        import win32gui
        import win32con
        hwnd = win32gui.GetForegroundWindow()
        win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        return True
    except Exception:
        try:
            import pyautogui
            pyautogui.FAILSAFE = False
            pyautogui.hotkey("alt", "f4")
            return True
        except Exception:
            return False


def switch_window_alt_tab(steps: int = 1) -> bool:
    """Cycle through open windows using Alt+Tab."""
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.keyDown("alt")
        for _ in range(steps):
            pyautogui.press("tab")
            time.sleep(0.1)
        pyautogui.keyUp("alt")
        return True
    except Exception:
        return False


def show_desktop() -> bool:
    """Show the Windows desktop (Win+D)."""
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.hotkey("win", "d")
        return True
    except Exception:
        return False


def snap_window(direction: str) -> bool:
    """Snap active window to left/right/up/down (Win+Arrow)."""
    key_map = {"left": "left", "right": "right", "up": "up", "down": "down",
               "top": "up", "bottom": "down", "half": "left"}
    key = key_map.get(direction.lower(), "left")
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.hotkey("win", key)
        return True
    except Exception:
        return False


# ─── Power Management ─────────────────────────────────────────────────────────

def lock_screen() -> bool:
    """Lock the Windows screen."""
    try:
        subprocess.Popen(["rundll32.exe", "user32.dll,LockWorkStation"])
        return True
    except Exception:
        return False


def sleep_system() -> bool:
    """Put the system to sleep."""
    try:
        subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0", "1", "0"])
        return True
    except Exception:
        return False


def shutdown_system(delay_seconds: int = 30) -> bool:
    """Schedule a shutdown (cancellable with ``shutdown /a``)."""
    try:
        subprocess.run(["shutdown", "/s", "/t", str(delay_seconds)])
        return True
    except Exception:
        return False


def restart_system(delay_seconds: int = 30) -> bool:
    """Schedule a restart (cancellable with ``shutdown /a``)."""
    try:
        subprocess.run(["shutdown", "/r", "/t", str(delay_seconds)])
        return True
    except Exception:
        return False


def cancel_shutdown() -> bool:
    """Cancel a pending shutdown."""
    try:
        subprocess.run(["shutdown", "/a"])
        return True
    except Exception:
        return False


# ─── Clipboard ────────────────────────────────────────────────────────────────

def get_clipboard() -> str:
    """Get current clipboard content."""
    try:
        import pyperclip
        return pyperclip.paste()
    except Exception:
        return ""


def set_clipboard(text: str) -> bool:
    """Set clipboard content."""
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        return False


def clear_clipboard() -> bool:
    """Clear the clipboard."""
    try:
        import pyperclip
        pyperclip.copy("")
        return True
    except Exception:
        return False


# ─── Screenshot ───────────────────────────────────────────────────────────────

def take_screenshot(save_path: Optional[str] = None) -> str:
    """Take a screenshot and return the file path."""
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        if not save_path:
            save_path = os.path.join(os.path.expanduser("~"), "Desktop", f"screenshot_{int(time.time())}.png")
        img = pyautogui.screenshot()
        img.save(save_path)
        return save_path
    except Exception as e:
        automation_logger.warning(f"[SystemController] Screenshot error: {e}")
        return ""


# ─── Process Management ───────────────────────────────────────────────────────

def list_processes() -> List[Dict[str, Any]]:
    """List all running processes."""
    try:
        import psutil
        return [
            {"pid": p.pid, "name": p.info["name"], "cpu": p.info.get("cpu_percent") or 0.0, "mem": round(p.info.get("memory_percent") or 0.0, 1)}
            for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"])
            if p.info.get("name")
        ]
    except Exception:
        return []


def _own_process_tree() -> set[int]:
    """PIDs that must never be killed: JARVIS itself, its parents and children."""
    import psutil
    pids = {os.getpid()}
    try:
        me = psutil.Process(os.getpid())
        pids.update(p.pid for p in me.parents())
        pids.update(p.pid for p in me.children(recursive=True))
    except Exception:
        pass
    return pids


def _normalize_proc_name(name: str) -> str:
    n = name.strip().lower()
    return n[:-4] if n.endswith(".exe") else n


def find_processes(name_or_pid: str, protected: Optional[List[str]] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Exact (case-insensitive) executable-name or PID match.

    Returns (killable matches, reasons for skipped matches). Never matches by substring.
    """
    import psutil
    target = name_or_pid.strip()
    protected_names = {_normalize_proc_name(p) for p in (protected or [])}
    own = _own_process_tree()
    matches: List[Dict[str, Any]] = []
    skipped: List[str] = []
    want_pid = int(target) if target.isdigit() else None
    want_name = _normalize_proc_name(target)
    for proc in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            pname = proc.info["name"] or ""
            if want_pid is not None:
                if proc.pid != want_pid:
                    continue
            elif _normalize_proc_name(pname) != want_name:
                continue
            if proc.pid in own:
                skipped.append(f"{pname} (PID {proc.pid}) is JARVIS itself")
                continue
            if _normalize_proc_name(pname) in protected_names:
                skipped.append(f"{pname} (PID {proc.pid}) is a protected system process")
                continue
            mem = proc.info.get("memory_info")
            matches.append({"pid": proc.pid, "name": pname, "rss_mb": round(mem.rss / 1048576, 1) if mem else None})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return matches, skipped


def kill_pids(pids: List[int]) -> List[int]:
    """Terminate exactly these PIDs (gracefully, then kill). Returns PIDs that ended."""
    import psutil
    ended: List[int] = []
    procs = []
    for pid in pids:
        try:
            p = psutil.Process(pid)
            p.terminate()
            procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    gone, alive = psutil.wait_procs(procs, timeout=3)
    ended.extend(p.pid for p in gone)
    for p in alive:
        try:
            p.kill()
            ended.append(p.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return ended


# ─── System Info ─────────────────────────────────────────────────────────────

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    info: Dict[str, Any] = {}
    try:
        import psutil
        info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        info["cpu_cores"] = psutil.cpu_count()
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024**3), 1)
        info["ram_used_gb"] = round(mem.used / (1024**3), 1)
        info["ram_percent"] = mem.percent
        info["disk_percent"] = psutil.disk_usage("/").percent
        if psutil.sensors_battery():
            bat = psutil.sensors_battery()
            info["battery_percent"] = round(bat.percent, 1)
            info["battery_plugged"] = bat.power_plugged
    except Exception:
        pass
    try:
        import platform
        info["os"] = platform.system()
        info["os_version"] = platform.version()
        info["hostname"] = platform.node()
    except Exception:
        pass
    try:
        info["volume"] = get_volume()
        info["brightness"] = get_brightness()
    except Exception:
        pass
    return info


# ─── Network ─────────────────────────────────────────────────────────────────

def get_wifi_status() -> Dict[str, Any]:
    """Get WiFi connection status."""
    try:
        r = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True, text=True, timeout=5
        )
        lines = r.stdout.strip().splitlines()
        info: Dict[str, Any] = {}
        for line in lines:
            if ":" in line:
                key, _, val = line.partition(":")
                info[key.strip().lower()] = val.strip()
        return info
    except Exception:
        return {}


def toggle_wifi(enable: bool) -> bool:
    """Enable or disable WiFi adapter."""
    action = "enable" if enable else "disable"
    try:
        subprocess.run(
            ["netsh", "interface", "set", "interface", "Wi-Fi", action],
            capture_output=True, timeout=5
        )
        return True
    except Exception:
        return False


def toggle_bluetooth(enable: bool) -> bool:
    """Enable or disable Bluetooth via PowerShell."""
    try:
        ps_cmd = f"""
$bt = [System.Runtime.InteropServices.RuntimeEnvironment]::GetRuntimeDirectory()
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where {{ $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' }})[0]
$await = {{ param($op, $type) $asTask.MakeGenericMethod($type).Invoke($null, @($op)).GetAwaiter().GetResult() }}
[Windows.Devices.Radios.Radio,Windows.System.Devices,ContentType=WindowsRuntime] | Out-Null
[Windows.Devices.Radios.RadioAccessStatus,Windows.System.Devices,ContentType=WindowsRuntime] | Out-Null
$radios = & $await ([Windows.Devices.Radios.Radio]::RequestAccessAsync()) ([Windows.Devices.Radios.RadioAccessStatus])
$bt_radio = ([Windows.Devices.Radios.Radio]::GetRadiosAsync() | & $await -type ([System.Collections.Generic.IReadOnlyList[Windows.Devices.Radios.Radio]])) | Where-Object {{ $_.Kind -eq [Windows.Devices.Radios.RadioKind]::Bluetooth }}
if ($bt_radio) {{ $bt_radio.SetStateAsync([Windows.Devices.Radios.RadioState]::{("On" if enable else "Off")}) }}
"""
        subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True, timeout=10)
        return True
    except Exception:
        return False


# ─── Keyboard Shortcuts / System Hotkeys ─────────────────────────────────────

def send_system_hotkey(*keys: str) -> bool:
    """Send a global system keyboard shortcut."""
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.hotkey(*keys)
        return True
    except Exception:
        return False


def open_task_manager() -> bool:
    """Open Windows Task Manager."""
    try:
        subprocess.Popen(["taskmgr.exe"])
        return True
    except Exception:
        return False


def open_settings(page: str = "") -> bool:
    """Open Windows Settings (optionally to a specific page)."""
    try:
        if page:
            os.startfile(f"ms-settings:{page}")
        else:
            os.startfile("ms-settings:")
        return True
    except Exception:
        return False


def open_file_explorer(path: str = "") -> bool:
    """Open File Explorer at given path."""
    try:
        if path:
            subprocess.Popen(["explorer.exe", path])
        else:
            subprocess.Popen(["explorer.exe"])
        return True
    except Exception:
        return False


def empty_recycle_bin() -> bool:
    """Empty the Windows Recycle Bin."""
    try:
        import ctypes
        # Flags: SHERB_NOCONFIRMATION (0x1) | SHERB_NOPROGRESSUI (0x2) | SHERB_NOSOUND (0x4) = 7
        res = ctypes.windll.shell32.SHEmptyRecycleBinW(None, None, 7)
        return res == 0
    except Exception:
        try:
            subprocess.run(["powershell", "-Command", "Clear-RecycleBin -Force -ErrorAction SilentlyContinue"], capture_output=True)
            return True
        except Exception:
            return False


def set_wallpaper(image_path: str) -> bool:
    """Change the desktop wallpaper."""
    try:
        import ctypes
        ctypes.windll.user32.SystemParametersInfoW(20, 0, image_path, 3)
        return True
    except Exception:
        return False
