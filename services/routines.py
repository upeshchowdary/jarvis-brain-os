"""Routines / modes: user-defined YAML with variables and conditions.

Files are read from ``data/routines/*.yaml`` (yours) and ``routines/*.yaml`` (built-in
examples). Step kinds:

    - op: <gateway operation>   args: {...}        # e.g. app.open, browser.open_url, system.volume, window.manage
    - say: "text with {variables}"                 # spoken / shown to the user
    - timer: {minutes: 50, label: "Break time"}     # one-off reminder
    - wait: 2                                       # seconds
    - briefing: true                                # include the daily briefing
    - if: {battery_below: 20 | app_running: chrome | weekday: [mon, tue] | after: "18:00" | before: "09:00"}
      then: [ ...steps... ]
      else: [ ...steps... ]

All ``op`` steps are executed as ONE gateway plan (preview + at most one confirmation,
dry-run aware, audited).
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from automation.gateway import ActionRequest, GatewayResult, gateway
from core.paths import BASE_DIR, data_dir

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

_VAR = re.compile(r"\{([a-zA-Z_]\w*)\}")
_DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def routine_dirs() -> List[Path]:
    user = data_dir() / "routines"
    user.mkdir(parents=True, exist_ok=True)
    return [user, BASE_DIR / "routines"]


def load_all() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for d in reversed(routine_dirs()):  # user files override built-ins
        for path in sorted(d.glob("*.y*ml")):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            name = str(data.get("name") or path.stem.replace("_", " ")).lower()
            data["_path"] = str(path)
            out[name] = data
    return out


def find(name: str) -> Optional[Dict[str, Any]]:
    routines = load_all()
    key = name.strip().lower()
    for candidate in (key, key.removesuffix(" mode"), f"{key} mode", key.removesuffix(" routine"), f"{key} routine"):
        if candidate in routines:
            return routines[candidate]
    for k, v in routines.items():
        if _slug(k) == _slug(key):
            return v
    return None


def fill(value: Any, variables: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        return _VAR.sub(lambda m: str(variables.get(m.group(1), m.group(0))), value)
    if isinstance(value, dict):
        return {k: fill(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [fill(v, variables) for v in value]
    return value


def condition_met(cond: Dict[str, Any], status: Dict[str, Any], now: datetime) -> bool:
    if "battery_below" in cond:
        bat = status.get("battery_percent")
        if bat is None or bat >= float(cond["battery_below"]):
            return False
    if "app_running" in cond:
        if str(cond["app_running"]).lower().removesuffix(".exe") not in status.get("processes", set()):
            return False
    if "weekday" in cond:
        days = cond["weekday"] if isinstance(cond["weekday"], list) else [cond["weekday"]]
        if _DAYS[now.weekday()] not in [str(d)[:3].lower() for d in days]:
            return False
    if "after" in cond and now.strftime("%H:%M") < str(cond["after"]):
        return False
    if "before" in cond and now.strftime("%H:%M") >= str(cond["before"]):
        return False
    return True


def expand(steps: List[Dict[str, Any]], variables: Dict[str, Any], status: Dict[str, Any], now: datetime) -> List[Dict[str, Any]]:
    """Resolve variables and conditions into a flat list of concrete steps."""
    flat: List[Dict[str, Any]] = []
    for step in steps or []:
        step = dict(step)
        if "if" in step:
            branch = step.get("then", []) if condition_met(step["if"], status, now) else step.get("else", [])
            flat.extend(expand(branch, variables, status, now))
            continue
        flat.append(fill(step, variables))
    return flat


async def _status() -> Dict[str, Any]:
    import psutil

    from tools.builtin.info import collect_system_status

    info = await asyncio.to_thread(collect_system_status)
    info["processes"] = {(p.info.get("name") or "").lower().removesuffix(".exe") for p in psutil.process_iter(["name"])}
    return info


def plan(routine: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None, status: Optional[Dict[str, Any]] = None,
         now: Optional[datetime] = None) -> Tuple[List[ActionRequest], List[Dict[str, Any]]]:
    """Split a routine into gateway actions and local steps (say/timer/wait/briefing)."""
    variables = {**(routine.get("variables") or {}), **(overrides or {})}
    steps = expand(routine.get("steps", []), variables, status or {}, now or datetime.now())
    actions = [ActionRequest(s["op"], dict(s.get("args") or {}), s.get("description", "")) for s in steps if "op" in s]
    local = [s for s in steps if "op" not in s]
    return actions, local


async def run(name: str, session_id: str = "default", overrides: Optional[Dict[str, Any]] = None,
              dry_run: Optional[bool] = None, source: str = "routine") -> Tuple[Optional[GatewayResult], List[str]]:
    routine = find(name)
    if routine is None:
        return None, [f"I don't have a routine called '{name}'. Available: {', '.join(load_all()) or 'none'}."]
    actions, local = plan(routine, overrides, await _status())
    messages: List[str] = []
    result = None
    if actions:
        result = await gateway.execute(actions, session_id, dry_run=dry_run, source=source)
        if result.status not in ("executed", "dry_run"):
            return result, [result.message]
        messages.append(result.message)
    from core.config import settings

    if settings.DRY_RUN if dry_run is None else dry_run:
        extra = [f"(would also: {', '.join(next(iter(s)) for s in local)})"] if local else []
        return result, messages + extra
    for step in local:
        if "say" in step:
            messages.append(str(step["say"]))
        elif "wait" in step:
            await asyncio.sleep(min(float(step["wait"]), 30))
        elif "timer" in step:
            from services.store import store

            t = step["timer"]
            minutes = float(t.get("minutes", 25))
            await store.add_reminder(str(t.get("label", "Timer done")), time.time() + minutes * 60)
            messages.append(f"Timer set for {minutes:g} minutes.")
        elif "briefing" in step:
            from tools.builtin.productivity import build_briefing

            messages.append(await build_briefing())
    return result, messages
