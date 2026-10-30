"""Structural guarantees: OS operations are only reachable through the gateway; no shell strings."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ["app", "automation", "brain", "core", "memory", "tools", "vision", "voice", "services", "integrations"]


def _py_files():
    for d in SOURCE_DIRS:
        base = ROOT / d
        if base.exists():
            yield from base.rglob("*.py")
    yield from ROOT.glob("*.py")


def test_ops_are_private_to_the_gateway():
    offenders = []
    for path in _py_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("automation/ops/") or rel == "automation/gateway.py":
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"^\s*(?:from|import)\s+automation\.ops\b", text, re.M):
            offenders.append(rel)
    assert offenders == [], f"These modules bypass the gateway: {offenders}"


def test_no_shell_strings_anywhere():
    offenders = []
    for path in _py_files():
        text = path.read_text(encoding="utf-8")
        if re.search(r"shell\s*=\s*True|os\.system\(|create_subprocess_shell\(", text):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []


def test_action_router_has_no_shell_or_system_control():
    text = (ROOT / "automation/engine/action_router.py").read_text(encoding="utf-8")
    assert "run_command" not in text and "system_control" not in text and "send_message" not in text


def test_side_effect_tools_declare_it():
    from tools.registry import load_builtin_tools

    reg = load_builtin_tools()
    side = {t.name for t in reg.all() if t.side_effect}
    for must in ("power_control", "kill_process", "send_message", "compose_email", "computer_task", "open_app",
                 "set_volume", "run_named_command", "memory_remember", "automation_control"):
        assert must in side, must
    read_only = {t.name for t in reg.read_only()}
    assert {"web_search", "files_read", "screen_look", "get_time_date", "clarify"} <= read_only
