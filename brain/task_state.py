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
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Deque, Dict, List, Optional

from app.utils.logger import logger
from core.config import settings

_PLAN_STATUSES = ("todo", "doing", "done", "failed")
_MARKS = {"done": "✓", "failed": "✗"}
# How a step (or an action note) ended. "pending" = awaiting the user's confirmation.
STEP_MARKS = {"ok": "✓", "failed": "✗", "pending": "…", "dry_run": "(dry run)", "cancelled": "(cancelled)"}
# Statuses that keep a task current (the others end it: done / failed / shelved).
_CURRENT_STATUSES = ("active", "waiting_for_user")


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
    state: str = ""  # one of STEP_MARKS; "" = derived from ok (older rows)

    @property
    def mark(self) -> str:
        return STEP_MARKS.get(self.state) or ("✓" if self.ok else "✗")


@dataclass
class Task:
    id: str
    goal: str
    status: str  # active | waiting_for_user | shelved | done | failed
    channel: str
    plan: List[PlanItem] = field(default_factory=list)
    steps: List[TaskStep] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Task":
        """Tolerant of unknown keys (older/newer rows); raises KeyError only if id/goal are missing."""
        def pick(kind: type, raw: Dict[str, Any]) -> Any:
            names = {f.name for f in fields(kind)}
            return kind(**{k: v for k, v in raw.items() if k in names})

        return cls(
            id=str(d["id"]), goal=str(d["goal"]), status=d.get("status", "shelved"), channel=d.get("channel", "text"),
            plan=[pick(PlanItem, p) for p in d.get("plan", [])],
            steps=[pick(TaskStep, x) for x in d.get("steps", [])],
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
                lines.append(f" {s.mark} {s.tool}({s.args_summary}): {s.result_summary}")
        return "\n".join(lines)


class TaskTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: Optional[Task] = None
        self._last: Optional[Task] = None  # most recent task that is no longer current
        self._loop: Optional[asyncio.AbstractEventLoop] = None  # set by load(); lets worker threads save
        self._seq = 0  # bumped per scheduled save; an older snapshot never overwrites a newer one
        self._saved_seq: Dict[str, int] = {}
        self._save_lock: Optional[asyncio.Lock] = None
        self._pending: set = set()  # strong refs to in-flight saves

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
    # Mutations take an optional ``task_id`` guard: when given and it is not the current task (another turn
    # started a new one meanwhile), the call is a no-op, so concurrent turns never write into each other's task.
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

    def record_step(self, tool: str, args: Dict[str, Any], result_message: str, ok: bool, state: str = "",
                    task_id: Optional[str] = None) -> None:
        args_summary = _clip(", ".join(f"{k}={v}" for k, v in (args or {}).items()), 60)
        with self._lock:
            t = self._target(task_id)
            if not t:
                return
            t.steps.append(TaskStep(tool, args_summary, _clip(result_message, 100), bool(ok), time.time(),
                                    state if state in STEP_MARKS else ""))
            del t.steps[:-50]
            self._touch(t)

    def set_plan(self, subgoals: List[str], keep_status: bool = False, task_id: Optional[str] = None) -> None:
        """Replace the plan. With keep_status, items whose text matches an existing one keep their status."""
        def norm(x: str) -> str:
            return " ".join(x.lower().split())

        with self._lock:
            t = self._target(task_id)
            if not t:
                return
            old = {norm(p.subgoal): p.status for p in t.plan} if keep_status else {}
            t.plan = [PlanItem(s.strip(), old.get(norm(s), "todo")) for s in subgoals if s and s.strip()]
            self._touch(t)

    def mark_plan(self, index: int, status: str, task_id: Optional[str] = None) -> None:
        if status not in _PLAN_STATUSES:
            return
        with self._lock:
            t = self._target(task_id)
            if not t or not 0 <= index < len(t.plan):
                return
            t.plan[index].status = status
            self._touch(t)

    def set_status(self, status: str, task_id: Optional[str] = None) -> None:
        with self._lock:
            t = self._target(task_id)
            if not t:
                return
            t.status = status
            if status not in _CURRENT_STATUSES:  # done / failed / shelved: no longer the current task
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

        try:
            self._loop = asyncio.get_running_loop()
            rows = await db_manager.load_recent_tasks(limit=5)
        except Exception as exc:  # never block startup
            logger.warning(f"Could not load tasks: {exc}")
            return
        with self._lock:
            for data in rows:  # newest first
                try:
                    t = Task.from_dict(data)
                except Exception as exc:  # a bad row must not stop startup
                    logger.warning(f"Skipping unreadable task row: {exc}")
                    continue
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
    def _target(self, task_id: Optional[str]) -> Optional[Task]:
        t = self._current
        return t if t is not None and (task_id is None or t.id == task_id) else None

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
        self._seq += 1
        args = (t.to_dict(), self._seq)
        try:
            task = asyncio.get_running_loop().create_task(self._save(*args))
        except RuntimeError:
            if not (self._loop and self._loop.is_running()):
                return
            try:
                task = asyncio.run_coroutine_threadsafe(self._save(*args), self._loop)
            except Exception:
                return
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _save(self, data: Dict[str, Any], seq: int) -> None:
        from app.database.connection import db_manager

        if self._save_lock is None:
            self._save_lock = asyncio.Lock()
        async with self._save_lock:  # one write at a time, in order
            if self._saved_seq.get(data["id"], 0) > seq:
                return  # a newer snapshot already landed
            self._saved_seq[data["id"]] = seq
            await db_manager.save_task(data)

    async def flush(self) -> None:
        """Wait for scheduled saves to finish (tests / shutdown)."""
        while self._pending:
            futs = [p if isinstance(p, asyncio.Future) else asyncio.wrap_future(p) for p in list(self._pending)]
            await asyncio.gather(*futs, return_exceptions=True)


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
