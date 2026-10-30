"""ActionGateway — the single door every side effect must pass through.

Every action that changes something on the computer (power, processes, apps, UI
input, files, messages, shell, workflow replay, voice commands …) is registered
as an *operation* with a risk level and executed only via ``gateway.execute``.

The gateway:
  * classifies risk (LOW / MEDIUM / HIGH, configurable),
  * enforces dry-run (global ``settings.DRY_RUN`` or a per-request override),
  * requires confirmation (HIGH always, MEDIUM unless relaxed) and stores the
    pending action per session with a TTL,
  * writes an append-only audit log (``data/audit.jsonl``),
  * honours the emergency stop,
  * records undo information so the last file operation can be reverted.

Operation implementations live in the private package ``automation.ops`` and are
never imported directly by other code (enforced by ``tests/test_gateway_isolation.py``).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Union

from core.config import settings
from core.logging import logger
from core.paths import data_path


class Risk(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass
class OpResult:
    """What an operation returns."""

    success: bool
    message: str = ""
    data: Any = None
    undo: Optional[Dict[str, Any]] = None  # {"op": ..., "args": {...}} to revert this action

    @classmethod
    def ok(cls, message: str = "", data: Any = None, undo: Optional[Dict[str, Any]] = None) -> "OpResult":
        return cls(True, message, data, undo)

    @classmethod
    def fail(cls, message: str, data: Any = None) -> "OpResult":
        return cls(False, message, data)


@dataclass
class Preview:
    """Human readable description shown before confirmation. ``ok=False`` aborts early."""

    text: str
    ok: bool = True
    data: Any = None


RiskSpec = Union[Risk, Callable[[Dict[str, Any]], Risk]]


@dataclass
class Operation:
    name: str
    risk: RiskSpec
    func: Callable[..., Awaitable[OpResult]]
    description: str = ""
    preview: Optional[Callable[[Dict[str, Any]], Awaitable[Preview] | Preview]] = None

    def risk_for(self, args: Dict[str, Any]) -> Risk:
        return self.risk(args) if callable(self.risk) else self.risk


@dataclass
class ActionRequest:
    op: str
    args: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"op": self.op, "args": self.args, "description": self.description}


@dataclass
class GatewayResult:
    # executed | needs_confirmation | dry_run | blocked | failed | cancelled | expired
    status: str
    message: str
    risk: str = "low"
    pending_id: Optional[str] = None
    preview: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.status in ("executed", "dry_run")

    @property
    def needs_confirmation(self) -> bool:
        return self.status == "needs_confirmation"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "success": self.success,
            "message": self.message,
            "risk": self.risk,
            "pending_id": self.pending_id,
            "preview": self.preview,
            "steps": self.steps,
        }


@dataclass
class PendingAction:
    pending_id: str
    session_id: str
    actions: List[ActionRequest]
    risk: Risk
    preview: str
    source: str
    created_at: float
    expires_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pending_id": self.pending_id,
            "session_id": self.session_id,
            "risk": self.risk.label,
            "preview": self.preview,
            "source": self.source,
            "actions": [a.to_dict() for a in self.actions],
            "expires_in": max(0.0, round(self.expires_at - time.time(), 1)),
        }


# Risk overrides, e.g. {"ui.type": "low"} in data/risk_policy.json lets the user relax a rule.
def _load_policy_overrides() -> Dict[str, Risk]:
    path = data_path("risk_policy.json")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: Dict[str, Risk] = {}
    for op, level in raw.items():
        try:
            out[str(op)] = Risk[str(level).upper()]
        except KeyError:
            logger.warning(f"[Gateway] Ignoring invalid risk override {op}={level}")
    return out


def _short(value: Any, limit: int = 200) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    if isinstance(value, dict):
        return {k: _short(v, limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_short(v, limit) for v in list(value)[:20]]
    return value


class ActionGateway:
    """Classifies, confirms, executes and audits every side-effectful action."""

    # Operations that are *always* HIGH, regardless of overrides (the policy can only relax MEDIUM).
    NEVER_RELAX = Risk.HIGH

    def __init__(self) -> None:
        self._ops: Dict[str, Operation] = {}
        self._pending: Dict[str, PendingAction] = {}
        self._stopped = threading.Event()
        self._stop_reason = ""
        self._audit_lock = threading.Lock()
        self._ops_loaded = False
        self._overrides: Optional[Dict[str, Risk]] = None
        self.listeners: List[Callable[[str, Dict[str, Any]], None]] = []

    # ── Registry ────────────────────────────────────────────────────────
    def register(self, op: Operation) -> Operation:
        self._ops[op.name] = op
        return op

    def operation(
        self,
        name: str,
        risk: RiskSpec,
        description: str = "",
        preview: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> Callable[[Callable[..., Awaitable[OpResult]]], Callable[..., Awaitable[OpResult]]]:
        def deco(func: Callable[..., Awaitable[OpResult]]) -> Callable[..., Awaitable[OpResult]]:
            self.register(Operation(name, risk, func, description or (func.__doc__ or "").strip(), preview))
            return func

        return deco

    def _ensure_ops(self) -> None:
        if not self._ops_loaded:
            self._ops_loaded = True
            import automation.ops  # noqa: F401  (registers all operations)

    def get(self, name: str) -> Optional[Operation]:
        self._ensure_ops()
        return self._ops.get(name)

    def list_operations(self) -> List[Dict[str, str]]:
        self._ensure_ops()
        return [
            {"name": o.name, "risk": (o.risk.label if isinstance(o.risk, Risk) else "dynamic"), "description": o.description}
            for o in sorted(self._ops.values(), key=lambda o: o.name)
        ]

    # ── Policy ──────────────────────────────────────────────────────────
    def classify(self, action: ActionRequest) -> Risk:
        op = self.get(action.op)
        if op is None:
            return Risk.HIGH
        risk = op.risk_for(action.args)
        if self._overrides is None:
            self._overrides = _load_policy_overrides()
        override = self._overrides.get(action.op)
        if override is not None and risk < self.NEVER_RELAX:
            risk = override
        return risk

    @staticmethod
    def needs_confirmation(risk: Risk) -> bool:
        if risk >= Risk.HIGH:
            return True
        if risk == Risk.MEDIUM:
            return settings.CONFIRM_MEDIUM_RISK
        return False

    # ── Emergency stop ──────────────────────────────────────────────────
    def emergency_stop(self, reason: str = "user request") -> None:
        self._stopped.set()
        self._stop_reason = reason
        cancelled = list(self._pending)
        self._pending.clear()
        self._audit("system", "emergency_stop", {"reason": reason, "cancelled": cancelled}, "high", "stop", True, reason)
        logger.warning(f"[Gateway] EMERGENCY STOP: {reason}")

    def resume(self) -> None:
        self._stopped.clear()
        self._stop_reason = ""
        self._audit("system", "resume", {}, "low", "resume", True, "")

    @property
    def is_stopped(self) -> bool:
        return self._stopped.is_set()

    # ── Pending confirmations ──────────────────────────────────────────
    def _purge_expired(self) -> None:
        now = time.time()
        for pid in [p for p, a in self._pending.items() if a.expires_at < now]:
            expired = self._pending.pop(pid)
            self._audit(expired.session_id, "pending", {"pending_id": pid}, expired.risk.label, "expired", False, "")

    def pending_for(self, session_id: str) -> Optional[PendingAction]:
        self._purge_expired()
        matches = [a for a in self._pending.values() if a.session_id == session_id]
        return max(matches, key=lambda a: a.created_at) if matches else None

    def list_pending(self) -> List[PendingAction]:
        self._purge_expired()
        return sorted(self._pending.values(), key=lambda a: a.created_at)

    def get_pending(self, pending_id: str) -> Optional[PendingAction]:
        self._purge_expired()
        return self._pending.get(pending_id)

    def cancel(self, pending_id: str, reason: str = "cancelled by user") -> bool:
        action = self._pending.pop(pending_id, None)
        if action:
            self._audit(action.session_id, "pending", {"pending_id": pending_id}, action.risk.label, "cancelled", True, reason)
            self._emit("pending_cancelled", {"pending_id": pending_id})
            return True
        return False

    def cancel_session(self, session_id: str) -> int:
        ids = [p for p, a in self._pending.items() if a.session_id == session_id]
        for pid in ids:
            self.cancel(pid)
        return len(ids)

    # ── Execution ───────────────────────────────────────────────────────
    async def execute(
        self,
        action: Union[ActionRequest, Sequence[ActionRequest]],
        session_id: str = "default",
        *,
        confirmed: bool = False,
        dry_run: Optional[bool] = None,
        source: str = "chat",
    ) -> GatewayResult:
        """Run one action or an ordered plan of actions through the safety policy."""
        actions = [action] if isinstance(action, ActionRequest) else list(action)
        if not actions:
            return GatewayResult("failed", "Nothing to do.")
        self._ensure_ops()

        unknown = [a.op for a in actions if a.op not in self._ops]
        if unknown:
            msg = f"Unknown operation(s): {', '.join(unknown)}"
            self._audit(session_id, ",".join(unknown), {}, "high", "blocked", False, msg, source)
            return GatewayResult("blocked", msg, risk="high")

        risk = max(self.classify(a) for a in actions)
        is_dry = settings.DRY_RUN if dry_run is None else dry_run

        if self.is_stopped:
            msg = f"Emergency stop is active ({self._stop_reason}). Say 'resume automation' to re-enable actions."
            for a in actions:
                self._audit(session_id, a.op, a.args, risk.label, "blocked", False, msg, source)
            return GatewayResult("blocked", msg, risk=risk.label)

        preview, ok = await self._build_preview(actions)
        if not ok:
            for a in actions:
                self._audit(session_id, a.op, a.args, risk.label, "failed", False, preview, source)
            return GatewayResult("failed", preview, risk=risk.label, preview=preview)

        if is_dry:
            for a in actions:
                self._audit(session_id, a.op, a.args, risk.label, "dry_run", True, "", source)
            note = " (would ask for confirmation)" if self.needs_confirmation(risk) else ""
            return GatewayResult(
                "dry_run",
                f"[DRY RUN] Nothing was executed{note}. Plan:\n{preview}",
                risk=risk.label,
                preview=preview,
                steps=[{**a.to_dict(), "risk": self.classify(a).label} for a in actions],
            )

        if self.needs_confirmation(risk) and not confirmed:
            return self._stash(actions, session_id, risk, preview, source)

        return await self._run(actions, session_id, risk, source, decision="confirmed" if confirmed else "executed")

    async def confirm(self, pending_id: str, session_id: Optional[str] = None) -> GatewayResult:
        self._purge_expired()
        pending = self._pending.get(pending_id)
        if pending is None:
            return GatewayResult("expired", "That confirmation has expired or was already handled.")
        if session_id is not None and pending.session_id != session_id:
            return GatewayResult("blocked", "That confirmation belongs to a different session.")
        self._pending.pop(pending_id, None)
        self._emit("pending_resolved", {"pending_id": pending_id})
        if self.is_stopped:
            return GatewayResult("blocked", "Emergency stop is active; the action was not executed.")
        return await self._run(pending.actions, pending.session_id, pending.risk, pending.source, decision="confirmed")

    def _stash(self, actions: List[ActionRequest], session_id: str, risk: Risk, preview: str, source: str) -> GatewayResult:
        # One pending action per session: a new request replaces the previous one.
        self.cancel_session(session_id)
        now = time.time()
        pid = uuid.uuid4().hex[:12]
        self._pending[pid] = PendingAction(
            pid, session_id, actions, risk, preview, source, now, now + settings.CONFIRMATION_TTL_SECONDS
        )
        for a in actions:
            self._audit(session_id, a.op, a.args, risk.label, "needs_confirmation", True, "", source)
        self._emit("pending_created", self._pending[pid].to_dict())
        return GatewayResult(
            "needs_confirmation",
            f"⚠️ This needs your confirmation ({risk.label.upper()} risk):\n{preview}\n\nSay **yes** to confirm or anything else to cancel.",
            risk=risk.label,
            pending_id=pid,
            preview=preview,
        )

    async def _build_preview(self, actions: List[ActionRequest]) -> tuple[str, bool]:
        lines: List[str] = []
        for i, a in enumerate(actions, 1):
            op = self._ops[a.op]
            text = a.description or op.description or a.op
            if op.preview is not None:
                try:
                    res = op.preview(a.args)
                    if inspect.isawaitable(res):
                        res = await res
                    if not res.ok:
                        return res.text, False
                    text = res.text
                except Exception as exc:  # preview must never crash the request
                    logger.warning(f"[Gateway] preview for {a.op} failed: {exc}")
            prefix = f"{i}. " if len(actions) > 1 else "• "
            lines.append(f"{prefix}{text}")
        return "\n".join(lines), True

    async def _run(self, actions: List[ActionRequest], session_id: str, risk: Risk, source: str, decision: str) -> GatewayResult:
        steps: List[Dict[str, Any]] = []
        messages: List[str] = []
        for a in actions:
            if self.is_stopped:
                messages.append("Stopped by emergency stop.")
                break
            op = self._ops[a.op]
            started = time.perf_counter()
            try:
                res = await op.func(**a.args)
                if not isinstance(res, OpResult):
                    res = OpResult(bool(res.get("success")), str(res.get("message", ""))) if isinstance(res, dict) else OpResult.ok(str(res))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(f"[Gateway] {a.op} raised")
                res = OpResult.fail(f"{a.op} failed: {exc}")
            elapsed = round((time.perf_counter() - started) * 1000, 1)
            self._audit(session_id, a.op, a.args, self.classify(a).label, decision, res.success, res.message, source, res.undo, elapsed)
            steps.append({**a.to_dict(), "success": res.success, "message": res.message, "data": _short(res.data), "elapsed_ms": elapsed})
            if res.message:
                messages.append(res.message)
            if not res.success:
                return GatewayResult("failed", "\n".join(messages) or f"{a.op} failed.", risk=risk.label, steps=steps)
        return GatewayResult("executed", "\n".join(messages) or "Done.", risk=risk.label, steps=steps)

    # ── Undo ────────────────────────────────────────────────────────────
    def last_undoable(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        undone = set()
        for entry in reversed(self.read_audit(limit=2000)):
            if entry.get("decision") == "undo" and entry.get("undo_of"):
                undone.add(entry["undo_of"])
            if entry.get("undo") and entry.get("success") and entry.get("id") not in undone:
                if session_id is None or entry.get("session") == session_id:
                    return entry
        return None

    async def undo_last(self, session_id: str = "default", *, confirmed: bool = False) -> GatewayResult:
        entry = self.last_undoable(None)
        if entry is None:
            return GatewayResult("failed", "There is no recent file operation I can undo.")
        undo = entry["undo"]
        request = ActionRequest(undo["op"], undo.get("args", {}), f"Undo: {entry.get('message') or entry.get('op')}")
        result = await self.execute(request, session_id, confirmed=confirmed, source="undo")
        if result.status == "executed":
            self._audit(session_id, "undo", {"of": entry["id"]}, "medium", "undo", True, result.message, "undo", undo_of=entry["id"])
        return result

    # ── Audit log ───────────────────────────────────────────────────────
    @property
    def audit_path(self):
        return data_path("audit.jsonl")

    def _audit(
        self,
        session_id: str,
        op: str,
        args: Dict[str, Any],
        risk: str,
        decision: str,
        success: bool,
        message: str,
        source: str = "chat",
        undo: Optional[Dict[str, Any]] = None,
        elapsed_ms: Optional[float] = None,
        undo_of: Optional[str] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "id": uuid.uuid4().hex[:12],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "session": session_id,
            "source": source,
            "op": op,
            "target": _short(args),
            "risk": risk,
            "decision": decision,
            "success": success,
            "message": _short(message, 400),
        }
        if undo:
            entry["undo"] = undo
        if undo_of:
            entry["undo_of"] = undo_of
        if elapsed_ms is not None:
            entry["elapsed_ms"] = elapsed_ms
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with self._audit_lock:
            with open(self.audit_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        self._emit("audit", entry)

    def read_audit(self, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            with open(self.audit_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()[-limit:]
        except OSError:
            return []
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        for listener in list(self.listeners):
            try:
                listener(event, payload)
            except Exception:
                pass


gateway = ActionGateway()

__all__ = [
    "ActionGateway",
    "ActionRequest",
    "GatewayResult",
    "OpResult",
    "Operation",
    "PendingAction",
    "Preview",
    "Risk",
    "gateway",
]
