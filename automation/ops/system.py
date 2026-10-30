"""System-control operations registered with the ActionGateway.

LOW    volume, brightness, window management, lock screen, screenshots, read-only info
HIGH   shutdown, restart, sleep, kill process, empty recycle bin, wifi/bluetooth off,
       changing settings
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict

from automation.gateway import OpResult, Preview, Risk, gateway
from automation.ops import system_impl as impl
from core.config import settings

_IS_WINDOWS = sys.platform == "win32"


def _clamp(value: Any, lo: int = 0, hi: int = 100, default: int = 50) -> int:
    try:
        return max(lo, min(hi, int(float(value))))
    except (TypeError, ValueError):
        return default


async def _run(func, *args) -> Any:
    return await asyncio.to_thread(func, *args)


def _require_windows() -> OpResult | None:
    if not _IS_WINDOWS:
        return OpResult.fail("This system action is only available on Windows.")
    return None


# ── Audio ────────────────────────────────────────────────────────────────

@gateway.operation("system.volume", Risk.LOW, "Change the system volume")
async def volume(action: str = "set", level: Any = None, step: Any = 10) -> OpResult:
    if (err := _require_windows()):
        return err
    action = action.lower()
    if action == "set":
        lvl = _clamp(level)
        ok = await _run(impl.set_volume, lvl)
        return OpResult(ok, f"Volume set to {lvl}%." if ok else "Couldn't change the volume.")
    if action in ("up", "down"):
        current = await _run(impl.get_volume)
        delta = _clamp(step, 1, 100, 10)
        target = _clamp(current + (delta if action == "up" else -delta))
        ok = await _run(impl.set_volume, target)
        if not ok:  # media-key fallback
            ok = await _run(impl.set_volume_via_keys, action, max(1, delta // 2))
        return OpResult(ok, f"Volume {'up' if action == 'up' else 'down'} to {target}%." if ok else "Couldn't change the volume.")
    if action in ("mute", "unmute", "toggle_mute"):
        ok = await _run(impl.set_volume_via_keys, "mute")
        return OpResult(ok, "Toggled mute." if ok else "Couldn't toggle mute.")
    if action == "get":
        vol = await _run(impl.get_volume)
        return OpResult.ok(f"Volume is at {vol}%.", data=vol)
    return OpResult.fail(f"Unknown volume action '{action}'.")


@gateway.operation("system.media", Risk.LOW, "Media playback keys")
async def media(action: str = "play_pause") -> OpResult:
    if (err := _require_windows()):
        return err
    keys = {"play_pause": "playpause", "play": "playpause", "pause": "playpause", "next": "nexttrack", "previous": "prevtrack"}
    key = keys.get(action.lower())
    if not key:
        return OpResult.fail(f"Unknown media action '{action}'.")
    ok = await _run(impl.send_system_hotkey, key)
    return OpResult(ok, {"next": "Next track.", "previous": "Previous track."}.get(action, "Toggled playback."))


# ── Brightness ───────────────────────────────────────────────────────────

@gateway.operation("system.brightness", Risk.LOW, "Change the screen brightness")
async def brightness(action: str = "set", level: Any = None, step: Any = 10) -> OpResult:
    if (err := _require_windows()):
        return err
    action = action.lower()
    if action == "set":
        lvl = _clamp(level)
        ok = await _run(impl.set_brightness, lvl)
        return OpResult(ok, f"Brightness set to {lvl}%." if ok else "Couldn't change brightness (external monitors may not support it).")
    if action in ("up", "down"):
        ok = await _run(impl.adjust_brightness, action, _clamp(step, 1, 100, 10))
        now = await _run(impl.get_brightness)
        return OpResult(ok, f"Brightness {action} to {now}%." if ok and now >= 0 else ("Brightness adjusted." if ok else "Couldn't change brightness."))
    if action == "get":
        now = await _run(impl.get_brightness)
        return OpResult.ok(f"Brightness is at {now}%." if now >= 0 else "Brightness isn't readable on this display.", data=now)
    return OpResult.fail(f"Unknown brightness action '{action}'.")


# ── Windows ──────────────────────────────────────────────────────────────

@gateway.operation("window.manage", Risk.LOW, "Minimize / maximize / restore / close / snap / switch windows")
async def window(action: str, target: str = "", direction: str = "left") -> OpResult:
    if (err := _require_windows()):
        return err
    action = action.lower()
    tgt = target.strip()
    if action in ("minimize", "maximize", "restore", "close", "focus"):
        if tgt and tgt.lower() not in ("this", "current", "active", "window", "it"):
            ok = await _run(impl.window_action, tgt, action)
            what = f"'{tgt}'"
        else:
            func = {
                "minimize": impl.minimize_active_window,
                "maximize": impl.maximize_active_window,
                "close": impl.close_active_window,
                "restore": lambda: impl.send_system_hotkey("win", "up"),
                "focus": lambda: True,
            }[action]
            ok = await _run(func)
            what = "the active window"
        past = {"minimize": "Minimized", "maximize": "Maximized", "restore": "Restored", "close": "Closed", "focus": "Focused"}[action]
        return OpResult(ok, f"{past} {what}." if ok else f"Couldn't find a window matching {what}.")
    if action == "snap":
        ok = await _run(impl.snap_window, direction)
        return OpResult(ok, f"Snapped window {direction}.")
    if action == "show_desktop":
        ok = await _run(impl.show_desktop)
        return OpResult(ok, "Showing the desktop.")
    if action == "switch":
        ok = await _run(impl.switch_window_alt_tab, 1)
        return OpResult(ok, "Switched window.")
    if action == "list":
        wins = await _run(impl.get_all_windows)
        titles = [w["title"][:70] for w in wins[:20]]
        return OpResult.ok("Open windows:\n" + "\n".join(f"• {t}" for t in titles) if titles else "No open windows found.", data=wins)
    return OpResult.fail(f"Unknown window action '{action}'.")


# ── Power ────────────────────────────────────────────────────────────────

@gateway.operation("system.lock", Risk.LOW, "Lock the screen")
async def lock() -> OpResult:
    if (err := _require_windows()):
        return err
    ok = await _run(impl.lock_screen)
    return OpResult(ok, "Locked the screen." if ok else "Couldn't lock the screen.")


def _power_preview(args: Dict[str, Any]) -> Preview:
    action = args.get("action", "")
    delay = settings.POWER_ACTION_DELAY_SECONDS
    text = {
        "shutdown": f"Shut down this PC in {delay} seconds (you can say 'cancel shutdown').",
        "restart": f"Restart this PC in {delay} seconds (you can say 'cancel shutdown').",
        "sleep": "Put this PC to sleep now.",
        "hibernate": "Hibernate this PC now.",
        "logoff": "Sign out of Windows now (unsaved work will be lost).",
    }.get(action)
    return Preview(text, ok=True) if text else Preview(f"Unknown power action '{action}'.", ok=False)


@gateway.operation("system.power", Risk.HIGH, "Shutdown / restart / sleep the PC", preview=_power_preview)
async def power(action: str) -> OpResult:
    if (err := _require_windows()):
        return err
    delay = settings.POWER_ACTION_DELAY_SECONDS
    action = action.lower()
    if action == "shutdown":
        ok = await _run(impl.shutdown_system, delay)
        return OpResult(ok, f"Shutdown scheduled in {delay} seconds. Say 'cancel shutdown' to abort.")
    if action == "restart":
        ok = await _run(impl.restart_system, delay)
        return OpResult(ok, f"Restart scheduled in {delay} seconds. Say 'cancel shutdown' to abort.")
    if action == "sleep":
        ok = await _run(impl.sleep_system)
        return OpResult(ok, "Going to sleep.")
    if action == "hibernate":
        ok = await _run(lambda: impl.subprocess.run(["shutdown", "/h"], capture_output=True).returncode == 0)
        return OpResult(ok, "Hibernating.")
    if action == "logoff":
        ok = await _run(lambda: impl.subprocess.run(["shutdown", "/l"], capture_output=True).returncode == 0)
        return OpResult(ok, "Signing out.")
    return OpResult.fail(f"Unknown power action '{action}'.")


@gateway.operation("system.cancel_shutdown", Risk.LOW, "Cancel a scheduled shutdown/restart")
async def cancel_shutdown() -> OpResult:
    if (err := _require_windows()):
        return err
    ok = await _run(impl.cancel_shutdown)
    return OpResult(ok, "Cancelled the scheduled shutdown." if ok else "There was no shutdown to cancel.")


# ── Processes ────────────────────────────────────────────────────────────

def _kill_preview(args: Dict[str, Any]) -> Preview:
    target = str(args.get("target", "")).strip()
    if not target:
        return Preview("Which process should I end?", ok=False)
    matches, skipped = impl.find_processes(target, settings.PROTECTED_PROCESSES)
    if not matches:
        reason = ("; ".join(skipped)) if skipped else f"No running process is named exactly '{target}'."
        return Preview(f"Not ending anything: {reason}", ok=False)
    lines = [f"End {len(matches)} process(es) named '{target}':"]
    lines += [f"   - {m['name']} (PID {m['pid']}, {m['rss_mb']} MB)" for m in matches[:15]]
    if skipped:
        lines.append(f"   (skipping: {'; '.join(skipped[:3])})")
    return Preview("\n".join(lines), data=matches)


@gateway.operation("process.kill", Risk.HIGH, "End a process by exact name or PID", preview=_kill_preview)
async def kill_process(target: str) -> OpResult:
    matches, skipped = await asyncio.to_thread(impl.find_processes, target, settings.PROTECTED_PROCESSES)
    if not matches:
        return OpResult.fail("; ".join(skipped) or f"No running process is named exactly '{target}'.")
    ended = await asyncio.to_thread(impl.kill_pids, [m["pid"] for m in matches])
    return OpResult(bool(ended), f"Ended {len(ended)} process(es) named '{target}'." if ended else "Couldn't end the process (access denied?).", data=ended)


@gateway.operation("process.list", Risk.LOW, "List top processes by CPU")
async def list_processes(limit: int = 10) -> OpResult:
    procs = await _run(impl.list_processes)
    top = sorted(procs, key=lambda p: (p.get("cpu") or 0, p.get("mem") or 0), reverse=True)[: int(limit)]
    lines = [f"• {p['name']} (PID {p['pid']}) — CPU {p['cpu']:.1f}%, RAM {p['mem']}%" for p in top]
    return OpResult.ok("Top processes:\n" + "\n".join(lines), data=top)


# ── Utilities ────────────────────────────────────────────────────────────

@gateway.operation("system.screenshot", Risk.LOW, "Save a screenshot to disk")
async def screenshot(path: str = "") -> OpResult:
    saved = await _run(impl.take_screenshot, path or None)
    return OpResult(bool(saved), f"Screenshot saved to {saved}." if saved else "Screenshot failed.", data=saved)


@gateway.operation("system.open_tool", Risk.LOW, "Open Task Manager, Settings or File Explorer")
async def open_tool(tool: str, page: str = "") -> OpResult:
    if (err := _require_windows()):
        return err
    tool = tool.lower()
    func = {"task_manager": impl.open_task_manager, "settings": lambda: impl.open_settings(page),
            "file_explorer": lambda: impl.open_file_explorer(page)}.get(tool)
    if func is None:
        return OpResult.fail(f"Unknown tool '{tool}'.")
    ok = await _run(func)
    return OpResult(ok, f"Opened {tool.replace('_', ' ').title()}.")


@gateway.operation("system.empty_recycle_bin", Risk.HIGH, "Permanently empty the Recycle Bin")
async def empty_recycle_bin() -> OpResult:
    if (err := _require_windows()):
        return err
    ok = await _run(impl.empty_recycle_bin)
    return OpResult(ok, "Emptied the Recycle Bin." if ok else "Couldn't empty the Recycle Bin.")


def _radio_risk(args: Dict[str, Any]) -> Risk:
    return Risk.LOW if args.get("enable") else Risk.HIGH


@gateway.operation("network.wifi", _radio_risk, "Turn Wi-Fi on/off")
async def wifi(enable: bool) -> OpResult:
    if (err := _require_windows()):
        return err
    ok = await _run(impl.toggle_wifi, bool(enable))
    return OpResult(ok, f"Wi-Fi turned {'on' if enable else 'off'}." if ok else "Couldn't change Wi-Fi (admin rights may be required).")


@gateway.operation("network.bluetooth", _radio_risk, "Turn Bluetooth on/off")
async def bluetooth(enable: bool) -> OpResult:
    if (err := _require_windows()):
        return err
    ok = await _run(impl.toggle_bluetooth, bool(enable))
    return OpResult(ok, f"Bluetooth turned {'on' if enable else 'off'}." if ok else "Couldn't change Bluetooth.")


@gateway.operation("system.set_wallpaper", Risk.HIGH, "Change the desktop wallpaper (a settings change)")
async def set_wallpaper(path: str) -> OpResult:
    if (err := _require_windows()):
        return err
    ok = await _run(impl.set_wallpaper, path)
    return OpResult(ok, "Wallpaper changed." if ok else "Couldn't change the wallpaper.")


@gateway.operation("clipboard.set", Risk.MEDIUM, "Replace the clipboard contents")
async def clipboard_set(text: str) -> OpResult:
    ok = await _run(impl.set_clipboard, text)
    return OpResult(ok, "Copied to the clipboard.")
