"""PC-wide current task (one user, shared by voice / console / API) and a cross-channel turn buffer.

The tracker is deliberately in-memory first: ``current()`` runs on every message and does no I/O.
Mutations schedule a fire-and-forget save so a restart can pick the task back up.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Optional

from app.utils.logger import logger
from core.config import settings

_PLAN_STATUSES = ("todo", "doing", "done", "failed")
_MARKS = {"done": "✓", "failed": "✗"}


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


@dataclass
class PlanItem:
    subgoal: str
    status: str = "todo"  # todo | doing | done | failed


@dataclass
class TaskStep:
    tool: str
    args_summary: str
    result_summary: str
    ok: bool
    at: float


@dataclass
class Task:
    id: str
    goal: str
    status: str  # active | shelved | done | failed
    channel: str
    plan: List[PlanItem] = field(default_factory=list)
    steps: List[TaskStep] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Task":
        return cls(
            id=d["id"], goal=d["goal"], status=d.get("status", "shelved"), channel=d.get("channel", "text"),
            plan=[PlanItem(**p) for p in d.get("plan", [])],
            steps=[TaskStep(**s) for s in d.get("steps", [])],
            created_at=d.get("created_at", 0.0), updated_at=d.get("updated_at", 0.0),
        )

    def summary(self, max_steps: int = 3) -> str:
        """Compact plain text for LLM prompts: goal, status, plan (✓ done, ✗ failed, → next), last steps."""
        lines = [f"Task: {_clip(self.goal, 200)} [{self.status}]"]
        if self.plan:
            nxt = next((i for i, p in enumerate(self.plan) if p.status in ("todo", "doing")), -1)
            lines.append("Plan:")
            for i, p in enumerate(self.plan):
                mark = _MARKS.get(p.status) or ("→" if i == nxt else " ")
                lines.append(f" {mark} {i + 1}. {p.subgoal}")
        if self.steps and max_steps > 0:
            lines.append("Recent steps:")
            for s in self.steps[-max_steps:]:
                lines.append(f" {'✓' if s.ok else '✗'} {s.tool}({s.args_summary}): {s.result_summary}")
        return "\n".join(lines)


class TaskTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: Optional[Task] = None
        self._last: Optional[Task] = None  # most recent task that is no longer current
        self._loop: Optional[asyncio.AbstractEventLoop] = None  # set by load(); lets worker threads save

    # -- reads (no I/O) -------------------------------------------------
    def current(self) -> Optional[Task]:
        with self._lock:
            t = self._current
            if t and time.time() - t.updated_at > settings.TASK_IDLE_SECONDS:
                self._shelve_locked(t)
                self._schedule_save(t)
                return None
            return t

    # -- mutations ------------------------------------------------------
    def start(self, goal: str, channel: str) -> Task:
        with self._lock:
            if self._current:
                self._shelve_locked(self._current)
                self._schedule_save(self._last)
            now = time.time()
            t = Task(id=uuid.uuid4().hex[:12], goal=goal.strip(), status="active", channel=channel,
                     created_at=now, updated_at=now)
            self._current = t
            self._schedule_save(t)
            return t

    def record_step(self, tool: str, args: Dict[str, Any], result_message: str, ok: bool) -> None:
        args_summary = _clip(", ".join(f"{k}={v}" for k, v in (args or {}).items()), 60)
        with self._lock:
            t = self._current
            if not t:
                return
            t.steps.append(TaskStep(tool, args_summary, _clip(result_message, 100), bool(ok), time.time()))
            del t.steps[:-50]
            self._touch(t)

    def set_plan(self, subgoals: List[str]) -> None:
        with self._lock:
            t = self._current
            if not t:
                return
            t.plan = [PlanItem(s.strip()) for s in subgoals if s and s.strip()]
            self._touch(t)

    def mark_plan(self, index: int, status: str) -> None:
        if status not in _PLAN_STATUSES:
            return
        with self._lock:
            t = self._current
            if not t or not 0 <= index < len(t.plan):
                return
            t.plan[index].status = status
            self._touch(t)

    def set_status(self, status: str) -> None:
        with self._lock:
            t = self._current
            if not t:
                return
            t.status = status
            if status != "active":  # done / failed / shelved: no longer the current task
                self._current, self._last = None, t
            self._touch(t)

    def revive_last(self) -> Optional[Task]:
        """Bring back the most recently shelved task (e.g. the user says 'continue')."""
        with self._lock:
            t = self._last
            if not t or t.status != "shelved":
                return None
            if self._current:
                self._shelve_locked(self._current)
                self._schedule_save(self._last)
            t.status = "active"
            self._current, self._last = t, None
            self._touch(t)
            return t

    async def load(self) -> None:
        """Restore the newest unfinished task from the DB (current if recent enough, else last shelved)."""
        from app.database.connection import db_manager

        self._loop = asyncio.get_running_loop()
        try:
            rows = await db_manager.load_recent_tasks(limit=5)
        except Exception as exc:  # never block startup
            logger.warning(f"Could not load tasks: {exc}")
            return
        with self._lock:
            for data in rows:  # newest first
                t = Task.from_dict(data)
                if t.status in ("done", "failed"):
                    continue
                if time.time() - t.updated_at <= settings.TASK_IDLE_SECONDS:
                    t.status = "active"
                    self._current = t
                else:
                    t.status = "shelved"
                    self._last = t
                break

    # -- internals (call with the lock held) ----------------------------
    def _shelve_locked(self, t: Task) -> None:
        t.status = "shelved"
        self._last = t
        if self._current is t:
            self._current = None

    def _touch(self, t: Task) -> None:
        t.updated_at = time.time()
        self._schedule_save(t)

    def _schedule_save(self, t: Optional[Task]) -> None:
        """Fire-and-forget; safe from worker threads and when no loop exists (then it is skipped)."""
        if t is None:
            return
        from app.database.connection import db_manager

        data = t.to_dict()
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(db_manager.save_task(data))
            return
        except RuntimeError:
            pass
        if self._loop and self._loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(db_manager.save_task(data), self._loop)
            except Exception:
                pass


@dataclass
class Turn:
    channel: str
    session_id: str
    user: str
    assistant: str
    at: float


class RecentTurns:
    """Ring buffer of the latest turns across every channel."""

    def __init__(self, size: int = 50) -> None:
        self._turns: Deque[Turn] = deque(maxlen=size)
        self._lock = threading.Lock()

    def add(self, channel: str, session_id: str, user: str, assistant: str) -> None:
        with self._lock:
            self._turns.append(Turn(channel, session_id, user, assistant, time.time()))

    def recent(self, max_turns: int, window_s: float) -> List[Turn]:
        """Newest ``max_turns`` turns within ``window_s`` seconds, oldest first."""
        cutoff = time.time() - window_s
        with self._lock:
            fresh = [t for t in self._turns if t.at >= cutoff]
        return sorted(fresh, key=lambda t: t.at)[-max_turns:] if max_turns > 0 else []


tasks = TaskTracker()
recent_turns = RecentTurns()
