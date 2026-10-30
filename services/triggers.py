"""Workflow triggers: time, file created in a folder, app launched, USB drive connected.

Triggers are declared in each workflow YAML (``triggers:`` list). When one fires the
workflow is replayed through the gateway (source="trigger"), so its risk policy holds:
LOW-risk workflows run unattended, others create a pending approval on the dashboard.

    triggers:
      - {type: time, at: "09:00", days: [mon, tue, wed, thu, fri]}
      - {type: file_created, folder: "~/Downloads", pattern: "*.pdf"}
      - {type: app_launched, app: "zoom.exe"}
      - {type: usb_connected}
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from core.logging import logger

_DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def time_trigger_due(trigger: Dict[str, Any], now: datetime, last_fired: Optional[str]) -> bool:
    at = str(trigger.get("at", ""))
    try:
        hh, mm = (int(x) for x in at.split(":", 1))
    except ValueError:
        return False
    days = [d[:3].lower() for d in trigger.get("days", _DAYS)]
    stamp = now.strftime("%Y-%m-%d")
    return _DAYS[now.weekday()] in days and (now.hour, now.minute) == (hh, mm) and last_fired != stamp


class TriggerManager:
    def __init__(self, interval: float = 2.0) -> None:
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self._seen_files: Dict[str, Set[str]] = {}
        self._seen_procs: Set[str] = set()
        self._seen_drives: Set[str] = set()
        self._time_fired: Dict[str, str] = {}
        self._primed = False

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        self._task = None

    def _workflows(self) -> List[tuple[str, List[Dict[str, Any]]]]:
        from automation.workflow_recorder import workflow_recorder

        out = []
        for name in workflow_recorder.list_workflows():
            wf = workflow_recorder.load_workflow(name)
            if wf and wf.triggers:
                out.append((name, wf.triggers))
        return out

    def _snapshot(self) -> tuple[Set[str], Set[str]]:
        import psutil

        procs = {(p.info.get("name") or "").lower() for p in psutil.process_iter(["name"])}
        drives = {p.mountpoint for p in psutil.disk_partitions(all=False) if "removable" in (p.opts or "").lower()}
        return procs, drives

    def _new_files(self, key: str, folder: str, pattern: str) -> List[str]:
        path = Path(os.path.expanduser(folder))
        try:
            current = {f.name for f in path.iterdir() if f.is_file() and fnmatch.fnmatch(f.name.lower(), pattern.lower())}
        except OSError:
            return []
        seen = self._seen_files.get(key)
        self._seen_files[key] = current
        return [] if seen is None else sorted(current - seen)

    async def check_once(self, now: Optional[datetime] = None) -> List[str]:
        now = now or datetime.now()
        fired: List[str] = []
        workflows = await asyncio.to_thread(self._workflows)
        if not workflows:
            return fired
        procs, drives = await asyncio.to_thread(self._snapshot)
        new_procs, new_drives = procs - self._seen_procs, drives - self._seen_drives
        first = not self._primed
        self._seen_procs, self._seen_drives, self._primed = procs, drives, True
        for name, triggers in workflows:
            for i, trig in enumerate(triggers):
                kind = trig.get("type")
                key = f"{name}#{i}"
                hit = False
                if kind == "time":
                    hit = time_trigger_due(trig, now, self._time_fired.get(key))
                    if hit:
                        self._time_fired[key] = now.strftime("%Y-%m-%d")
                elif kind == "file_created":
                    hit = bool(self._new_files(key, trig.get("folder", "~/Downloads"), trig.get("pattern", "*")))
                elif kind == "app_launched" and not first:
                    hit = str(trig.get("app", "")).lower() in new_procs
                elif kind == "usb_connected" and not first:
                    hit = bool(new_drives)
                if hit:
                    fired.append(name)
                    await self._fire(name, trig)
        return fired

    async def _fire(self, name: str, trigger: Dict[str, Any]) -> None:
        from automation.workflow_player import workflow_player
        from services.notify import notify

        logger.info(f"[Triggers] {trigger.get('type')} → workflow '{name}'")
        res = await workflow_player.replay(name, trigger.get("params") or {}, session_id="triggers", source="trigger")
        await notify("⚙️ Workflow triggered", f"'{name}' ({trigger.get('type')}): {res.message[:160]}", speak=False, kind="trigger")

    async def _loop(self) -> None:
        while True:
            try:
                await self.check_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug(f"[Triggers] check failed: {exc}")
            await asyncio.sleep(self.interval)


trigger_manager = TriggerManager()
