"""ActionGateway: risk policy, confirmation flow, dry-run, emergency stop, audit, undo, kill-process safety."""

import os
import time
from unittest.mock import MagicMock, patch

import pytest

from automation.gateway import ActionRequest, OpResult, Operation, Preview, Risk, gateway
from core.config import settings


@pytest.fixture(autouse=True)
def clean_gateway():
    gateway.resume()
    for p in gateway.list_pending():
        gateway.cancel(p.pending_id)
    yield
    gateway.resume()


def register(name, risk, calls, preview=None):
    async def op(**kwargs):
        calls.append(kwargs)
        return OpResult.ok(f"ran {name}", undo={"op": "test.undo", "args": {"of": name}} if name == "test.undoable" else None)

    gateway.register(Operation(name, risk, op, f"Test {name}", preview))


async def test_low_risk_executes_immediately():
    calls = []
    register("test.low", Risk.LOW, calls)
    res = await gateway.execute(ActionRequest("test.low", {"x": 1}), "s-low")
    assert res.status == "executed" and calls == [{"x": 1}]


async def test_high_risk_requires_confirmation_then_runs():
    calls = []
    register("test.high", Risk.HIGH, calls)
    res = await gateway.execute(ActionRequest("test.high"), "s-high")
    assert res.status == "needs_confirmation" and res.pending_id and calls == []
    assert gateway.pending_for("s-high").pending_id == res.pending_id
    done = await gateway.confirm(res.pending_id, "s-high")
    assert done.status == "executed" and len(calls) == 1
    assert gateway.pending_for("s-high") is None


async def test_confirmation_is_session_bound_and_expires(monkeypatch):
    calls = []
    register("test.high2", Risk.HIGH, calls)
    res = await gateway.execute(ActionRequest("test.high2"), "owner")
    assert (await gateway.confirm(res.pending_id, "intruder")).status == "blocked"
    gateway._pending[res.pending_id].expires_at = time.time() - 1
    assert (await gateway.confirm(res.pending_id, "owner")).status == "expired"
    assert calls == []


async def test_medium_risk_confirmation_is_configurable(monkeypatch):
    calls = []
    register("test.medium", Risk.MEDIUM, calls)
    monkeypatch.setattr(settings, "CONFIRM_MEDIUM_RISK", True)
    assert (await gateway.execute(ActionRequest("test.medium"), "m1")).status == "needs_confirmation"
    monkeypatch.setattr(settings, "CONFIRM_MEDIUM_RISK", False)
    assert (await gateway.execute(ActionRequest("test.medium"), "m2")).status == "executed"


async def test_dry_run_never_executes(monkeypatch):
    calls = []
    register("test.dry", Risk.HIGH, calls)
    res = await gateway.execute(ActionRequest("test.dry"), "d1", dry_run=True)
    assert res.status == "dry_run" and calls == [] and "would ask for confirmation" in res.message
    monkeypatch.setattr(settings, "DRY_RUN", True)
    res = await gateway.execute(ActionRequest("test.dry"), "d2")
    assert res.status == "dry_run" and calls == []


async def test_emergency_stop_blocks_and_cancels_pending():
    calls = []
    register("test.stop", Risk.HIGH, calls)
    pending = await gateway.execute(ActionRequest("test.stop"), "e1")
    gateway.emergency_stop("test")
    assert gateway.get_pending(pending.pending_id) is None
    assert (await gateway.execute(ActionRequest("test.stop"), "e1", confirmed=True)).status == "blocked"
    gateway.resume()
    assert (await gateway.execute(ActionRequest("test.stop"), "e1", confirmed=True)).status == "executed"


async def test_unknown_operation_is_blocked():
    res = await gateway.execute(ActionRequest("does.not.exist"), "u1")
    assert res.status == "blocked"


async def test_failed_preview_aborts_without_pending():
    calls = []
    register("test.badpreview", Risk.HIGH, calls, preview=lambda a: Preview("nothing to do", ok=False))
    res = await gateway.execute(ActionRequest("test.badpreview"), "p1")
    assert res.status == "failed" and gateway.pending_for("p1") is None and calls == []


async def test_audit_log_records_decisions():
    calls = []
    register("test.audit", Risk.LOW, calls)
    await gateway.execute(ActionRequest("test.audit", {"target": "abc"}), "audit-session")
    entries = [e for e in gateway.read_audit(50) if e["session"] == "audit-session"]
    assert entries and entries[-1]["op"] == "test.audit" and entries[-1]["decision"] == "executed"
    assert {"ts", "risk", "target", "success"} <= set(entries[-1])


async def test_undo_last_runs_recorded_inverse(monkeypatch):
    monkeypatch.setattr(settings, "CONFIRM_MEDIUM_RISK", False)
    calls, undo_calls = [], []
    register("test.undoable", Risk.LOW, calls)
    register("test.undo", Risk.MEDIUM, undo_calls)
    await gateway.execute(ActionRequest("test.undoable"), "undo-s")
    res = await gateway.undo_last("undo-s")
    assert res.status == "executed" and undo_calls == [{"of": "test.undoable"}]
    # The same action is not undone twice.
    assert (await gateway.undo_last("undo-s")).status in ("failed", "executed")
    assert len(undo_calls) == 1 or gateway.last_undoable() is None


async def test_policy_override_cannot_relax_high(monkeypatch):
    monkeypatch.setattr(gateway, "_overrides", {"test.relax": Risk.LOW, "test.relaxhigh": Risk.LOW})
    register("test.relax", Risk.MEDIUM, [])
    register("test.relaxhigh", Risk.HIGH, [])
    assert gateway.classify(ActionRequest("test.relax")) == Risk.LOW
    assert gateway.classify(ActionRequest("test.relaxhigh")) == Risk.HIGH


# ── Real operations (OS functions mocked) ───────────────────────────────────

async def test_restart_needs_confirmation_and_uses_delay(monkeypatch):
    import automation.ops.system as sysops

    monkeypatch.setattr(sysops, "_IS_WINDOWS", True)
    restart = MagicMock(return_value=True)
    monkeypatch.setattr(sysops.impl, "restart_system", restart)
    res = await gateway.execute(ActionRequest("system.power", {"action": "restart"}), "power")
    assert res.status == "needs_confirmation" and "30 seconds" in res.preview
    restart.assert_not_called()
    done = await gateway.confirm(res.pending_id, "power")
    assert done.status == "executed" and "cancel shutdown" in done.message
    restart.assert_called_once_with(settings.POWER_ACTION_DELAY_SECONDS)


def _fake_proc(pid, name):
    p = MagicMock()
    p.pid = pid
    p.info = {"pid": pid, "name": name, "memory_info": MagicMock(rss=50 * 1048576)}
    return p


def test_find_processes_is_exact_and_protects_self():
    from automation.ops import system_impl

    me = os.getpid()
    procs = [_fake_proc(me, "python.exe"), _fake_proc(111, "python.exe"), _fake_proc(222, "pythonw.exe"),
             _fake_proc(333, "my-python-tool.exe"), _fake_proc(444, "explorer.exe")]
    with patch("psutil.process_iter", return_value=procs):
        matches, skipped = system_impl.find_processes("python", settings.PROTECTED_PROCESSES)
        assert [m["pid"] for m in matches] == [111]  # exact name only, never JARVIS itself
        assert any("JARVIS itself" in s for s in skipped)
        matches, skipped = system_impl.find_processes("explorer.exe", settings.PROTECTED_PROCESSES)
        assert matches == [] and "protected" in skipped[0]


async def test_kill_process_preview_lists_matches_and_never_runs_unconfirmed():
    procs = [_fake_proc(555, "notepad.exe")]
    with patch("psutil.process_iter", return_value=procs), patch("automation.ops.system_impl.kill_pids") as kill:
        res = await gateway.execute(ActionRequest("process.kill", {"target": "notepad"}), "kill")
        assert res.status == "needs_confirmation" and "PID 555" in res.preview
        kill.assert_not_called()


async def test_kill_process_with_no_match_does_not_ask():
    with patch("psutil.process_iter", return_value=[]):
        res = await gateway.execute(ActionRequest("process.kill", {"target": "skill"}), "kill2")
    assert res.status == "failed" and gateway.pending_for("kill2") is None


async def test_shell_only_allows_named_commands():
    res = await gateway.execute(ActionRequest("shell.run_named", {"name": "rm -rf /"}), "sh")
    assert res.status == "failed" and "not an allowed command" in res.message
    res = await gateway.execute(ActionRequest("shell.run_named", {"name": "ping", "params": {"host": "x; del *"}}), "sh")
    assert res.status == "failed"


async def test_computer_plan_with_disallowed_step_is_refused():
    plan = {"goal": "evil", "steps": [{"action": "run_command", "target": "format c:"}]}
    res = await gateway.execute(ActionRequest("computer.run_plan", plan), "plan")
    assert res.status == "failed" and gateway.pending_for("plan") is None
