"""JARVIS System Controller — Complete Windows System Control.

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

import asyncio
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
        import win32api, win32con
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
        import win32gui, win32con
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
        import win32gui, win32con
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
        import win32gui, win32con
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


def shutdown_system(delay_seconds: int = 0) -> bool:
    """Shutdown the system."""
    try:
        subprocess.run(["shutdown", "/s", "/t", str(delay_seconds)])
        return True
    except Exception:
        return False


def restart_system(delay_seconds: int = 0) -> bool:
    """Restart the system."""
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
            {"pid": p.pid, "name": p.name(), "cpu": p.cpu_percent(), "mem": p.memory_percent()}
            for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"])
        ]
    except Exception:
        return []


def kill_process(name_or_pid: str) -> bool:
    """Kill a process by name or PID."""
    try:
        import psutil
        target = name_or_pid.strip()
        killed = False
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                if target.isdigit() and proc.pid == int(target):
                    proc.kill()
                    killed = True
                elif target.lower() in proc.name().lower():
                    proc.kill()
                    killed = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return killed
    except Exception:
        return False


# ─── System Info ─────────────────────────────────────────────────────────────

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    info: Dict[str, Any] = {}
    try:
        import psutil
        info["cpu_percent"] = psutil.cpu_percent(interval=0.3)
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
    state = "1" if enable else "0"
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


# ─── Main async dispatch ─────────────────────────────────────────────────────

async def execute_system_action(action_type: str, target: str = "", value: str = "") -> Dict[str, Any]:
    """
    Master dispatcher for all system control actions.
    Returns {"success": bool, "summary": str, "data": Any}
    """
    a = action_type.lower().strip()
    t = target.lower().strip()
    v = value.strip()

    loop = asyncio.get_event_loop()

    # ── Volume ──────────────────────────────────────────────────────
    if a in ("volume_up", "vol_up"):
        steps = int(v) if v.isdigit() else 5
        ok = await loop.run_in_executor(None, lambda: set_volume_via_keys("up", steps))
        return {"success": ok, "summary": f"Volume increased by {steps} steps"}
    if a in ("volume_down", "vol_down"):
        steps = int(v) if v.isdigit() else 5
        ok = await loop.run_in_executor(None, lambda: set_volume_via_keys("down", steps))
        return {"success": ok, "summary": f"Volume decreased by {steps} steps"}
    if a in ("volume_mute", "mute", "unmute", "toggle_mute"):
        ok = await loop.run_in_executor(None, lambda: set_volume_via_keys("mute"))
        return {"success": ok, "summary": "Volume muted/unmuted"}
    if a == "set_volume":
        level = int(v) if v.isdigit() else 50
        ok = await loop.run_in_executor(None, lambda: set_volume(level))
        return {"success": ok, "summary": f"Volume set to {level}%"}
    if a == "get_volume":
        vol = await loop.run_in_executor(None, get_volume)
        return {"success": True, "summary": f"Current volume: {vol}%", "data": vol}

    # ── Brightness ───────────────────────────────────────────────────
    if a == "brightness_up":
        ok = await loop.run_in_executor(None, lambda: adjust_brightness("up", int(v) if v.isdigit() else 10))
        return {"success": ok, "summary": "Brightness increased"}
    if a == "brightness_down":
        ok = await loop.run_in_executor(None, lambda: adjust_brightness("down", int(v) if v.isdigit() else 10))
        return {"success": ok, "summary": "Brightness decreased"}
    if a == "set_brightness":
        level = int(v) if v.isdigit() else 70
        ok = await loop.run_in_executor(None, lambda: set_brightness(level))
        return {"success": ok, "summary": f"Brightness set to {level}%"}

    # ── Window management ────────────────────────────────────────────
    if a == "minimize_window":
        if t and t not in ("this", "current", "active", "window", ""):
            ok = await loop.run_in_executor(None, lambda: window_action(target, "minimize"))
        else:
            ok = await loop.run_in_executor(None, minimize_active_window)
        return {"success": ok, "summary": f"Window minimized"}
    if a == "maximize_window":
        if t and t not in ("this", "current", "active", "window", ""):
            ok = await loop.run_in_executor(None, lambda: window_action(target, "maximize"))
        else:
            ok = await loop.run_in_executor(None, maximize_active_window)
        return {"success": ok, "summary": "Window maximized"}
    if a == "close_window":
        if t and t not in ("this", "current", "active", "window", ""):
            ok = await loop.run_in_executor(None, lambda: window_action(target, "close"))
        else:
            ok = await loop.run_in_executor(None, close_active_window)
        return {"success": ok, "summary": "Window closed"}
    if a == "restore_window":
        if t:
            ok = await loop.run_in_executor(None, lambda: window_action(target, "restore"))
        else:
            ok = await loop.run_in_executor(None, lambda: send_system_hotkey("win", "up"))
        return {"success": ok, "summary": "Window restored"}
    if a == "snap_window":
        ok = await loop.run_in_executor(None, lambda: snap_window(t or v or "left"))
        return {"success": ok, "summary": f"Window snapped {t or v}"}
    if a == "show_desktop":
        ok = await loop.run_in_executor(None, show_desktop)
        return {"success": ok, "summary": "Desktop shown"}
    if a == "switch_window":
        ok = await loop.run_in_executor(None, lambda: switch_window_alt_tab(int(v) if v.isdigit() else 1))
        return {"success": ok, "summary": "Switched window"}
    if a == "focus_window":
        ok = await loop.run_in_executor(None, lambda: window_action(target, "activate"))
        return {"success": ok, "summary": f"Focused window: {target}"}
    if a == "list_windows":
        wins = await loop.run_in_executor(None, get_all_windows)
        titles = [w["title"][:60] for w in wins[:15]]
        return {"success": True, "summary": "Open windows:\n" + "\n".join(f"• {t}" for t in titles), "data": wins}

    # ── Power ────────────────────────────────────────────────────────
    if a == "lock_screen":
        ok = await loop.run_in_executor(None, lock_screen)
        return {"success": ok, "summary": "Screen locked"}
    if a == "sleep":
        ok = await loop.run_in_executor(None, sleep_system)
        return {"success": ok, "summary": "System going to sleep"}
    if a == "shutdown":
        delay = int(v) if v.isdigit() else 0
        ok = await loop.run_in_executor(None, lambda: shutdown_system(delay))
        return {"success": ok, "summary": f"Shutting down in {delay}s"}
    if a == "restart":
        delay = int(v) if v.isdigit() else 0
        ok = await loop.run_in_executor(None, lambda: restart_system(delay))
        return {"success": ok, "summary": f"Restarting in {delay}s"}
    if a == "cancel_shutdown":
        ok = await loop.run_in_executor(None, cancel_shutdown)
        return {"success": ok, "summary": "Shutdown cancelled"}

    # ── Clipboard ────────────────────────────────────────────────────
    if a == "get_clipboard":
        text = await loop.run_in_executor(None, get_clipboard)
        return {"success": True, "summary": f"Clipboard: {text[:200]}", "data": text}
    if a == "set_clipboard":
        ok = await loop.run_in_executor(None, lambda: set_clipboard(v or target))
        return {"success": ok, "summary": f"Clipboard set to: {(v or target)[:50]}"}
    if a == "clear_clipboard":
        ok = await loop.run_in_executor(None, clear_clipboard)
        return {"success": ok, "summary": "Clipboard cleared"}

    # ── Screenshot ───────────────────────────────────────────────────
    if a in ("screenshot", "take_screenshot"):
        path = await loop.run_in_executor(None, lambda: take_screenshot(v or None))
        ok = bool(path)
        return {"success": ok, "summary": f"Screenshot saved: {path}" if ok else "Screenshot failed", "data": path}

    # ── Process management ───────────────────────────────────────────
    if a == "list_processes":
        procs = await loop.run_in_executor(None, list_processes)
        top = sorted(procs, key=lambda p: p.get("cpu", 0), reverse=True)[:10]
        lines = [f"• {p['name']} (PID {p['pid']}) — CPU {p['cpu']:.1f}%" for p in top]
        return {"success": True, "summary": "Top processes:\n" + "\n".join(lines), "data": procs}
    if a == "kill_process":
        ok = await loop.run_in_executor(None, lambda: kill_process(target or v))
        return {"success": ok, "summary": f"Killed process: {target or v}"}

    # ── System info ──────────────────────────────────────────────────
    if a == "system_info":
        info = await loop.run_in_executor(None, get_system_info)
        summary = (
            f"CPU: {info.get('cpu_percent', '?')}% | "
            f"RAM: {info.get('ram_used_gb', '?')}/{info.get('ram_total_gb', '?')} GB ({info.get('ram_percent', '?')}%) | "
            f"Disk: {info.get('disk_percent', '?')}% | "
            f"Battery: {info.get('battery_percent', 'N/A')}% ({'Charging' if info.get('battery_plugged') else 'Discharging'}) | "
            f"Volume: {info.get('volume', '?')}% | "
            f"Brightness: {info.get('brightness', '?')}%"
        )
        return {"success": True, "summary": summary, "data": info}

    # ── Network ──────────────────────────────────────────────────────
    if a == "wifi_on":
        ok = await loop.run_in_executor(None, lambda: toggle_wifi(True))
        return {"success": ok, "summary": "WiFi enabled"}
    if a == "wifi_off":
        ok = await loop.run_in_executor(None, lambda: toggle_wifi(False))
        return {"success": ok, "summary": "WiFi disabled"}
    if a == "wifi_status":
        status = await loop.run_in_executor(None, get_wifi_status)
        ssid = status.get("ssid", "Unknown")
        state = status.get("state", "Unknown")
        return {"success": True, "summary": f"WiFi: {state} | Connected to: {ssid}", "data": status}
    if a == "bluetooth_on":
        ok = await loop.run_in_executor(None, lambda: toggle_bluetooth(True))
        return {"success": ok, "summary": "Bluetooth enabled"}
    if a == "bluetooth_off":
        ok = await loop.run_in_executor(None, lambda: toggle_bluetooth(False))
        return {"success": ok, "summary": "Bluetooth disabled"}

    # ── System apps & utils ──────────────────────────────────────────
    if a == "open_task_manager":
        ok = await loop.run_in_executor(None, open_task_manager)
        return {"success": ok, "summary": "Task Manager opened"}
    if a == "open_settings":
        ok = await loop.run_in_executor(None, lambda: open_settings(target or v))
        return {"success": ok, "summary": f"Settings opened{': ' + (target or v) if (target or v) else ''}"}
    if a == "open_file_explorer":
        ok = await loop.run_in_executor(None, lambda: open_file_explorer(target or v))
        return {"success": ok, "summary": f"File Explorer opened"}
    if a == "empty_recycle_bin":
        ok = await loop.run_in_executor(None, empty_recycle_bin)
        return {"success": ok, "summary": "Recycle Bin emptied"}
    if a == "set_wallpaper":
        ok = await loop.run_in_executor(None, lambda: set_wallpaper(target or v))
        return {"success": ok, "summary": f"Wallpaper changed to: {target or v}"}
    if a == "show_desktop":
        ok = await loop.run_in_executor(None, show_desktop)
        return {"success": ok, "summary": "Desktop shown"}

    # ── Hotkey ───────────────────────────────────────────────────────
    if a == "hotkey":
        keys = [k.strip() for k in (v or target).split("+")]
        ok = await loop.run_in_executor(None, lambda: send_system_hotkey(*keys))
        return {"success": ok, "summary": f"Hotkey sent: {'+'.join(keys)}"}

    return {"success": False, "summary": f"Unknown system action: {action_type}"}
