"""Allow-listed named commands (no free-form shell, ever).

The LLM planner cannot emit shell commands. When a command line is genuinely
needed it must be one of the named entries below; arguments are validated and
passed as an argument list with ``shell=False``.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import sys
from typing import Dict, List, Optional

from automation.gateway import OpResult, Preview, Risk, gateway

_HOST = re.compile(r"^[A-Za-z0-9.-]{1,253}$")
_SAFE_ARG = re.compile(r"^[\w.:/\\@+=, -]{0,200}$")

# name -> (argv template, description, risk)
ALLOWED_COMMANDS: Dict[str, tuple[List[str], str, Risk]] = {
    "ipconfig": (["ipconfig"] if sys.platform == "win32" else ["ip", "addr"], "Show network configuration", Risk.LOW),
    "ping": (["ping", "-n" if sys.platform == "win32" else "-c", "4", "{host}"], "Ping a host 4 times", Risk.LOW),
    "flush_dns": (["ipconfig", "/flushdns"], "Flush the DNS cache", Risk.HIGH),
    "git_status": (["git", "-C", "{path}", "status", "--short", "--branch"], "Show git status", Risk.LOW),
    "git_log_today": (["git", "-C", "{path}", "log", "--since=midnight", "--pretty=format:%h %an %s"], "List today's commits", Risk.LOW),
    "git_diff_stat": (["git", "-C", "{path}", "diff", "--stat"], "Summarize uncommitted changes", Risk.LOW),
    "pytest": ([sys.executable, "-m", "pytest", "-q", "--no-header", "-x", "{path}"], "Run the Python test suite", Risk.HIGH),
    "npm_test": (["npm", "test", "--prefix", "{path}"], "Run npm tests", Risk.HIGH),
}


def _build_argv(name: str, params: Dict[str, str]) -> tuple[Optional[List[str]], str]:
    if name not in ALLOWED_COMMANDS:
        return None, f"'{name}' is not an allowed command. Allowed: {', '.join(sorted(ALLOWED_COMMANDS))}."
    template, _, _ = ALLOWED_COMMANDS[name]
    argv: List[str] = []
    for part in template:
        if part.startswith("{") and part.endswith("}"):
            key = part[1:-1]
            value = str(params.get(key, "." if key == "path" else "")).strip()
            if key == "host" and not _HOST.match(value):
                return None, f"Invalid host '{value}'."
            if not _SAFE_ARG.match(value) or value.startswith("-"):
                return None, f"Invalid {key} '{value}'."
            argv.append(value)
        else:
            argv.append(part)
    return argv, ""


def _risk(args) -> Risk:
    entry = ALLOWED_COMMANDS.get(str(args.get("name", "")))
    return entry[2] if entry else Risk.HIGH


def _preview(args) -> Preview:
    argv, err = _build_argv(str(args.get("name", "")), dict(args.get("params") or {}))
    if argv is None:
        return Preview(err, ok=False)
    return Preview(f"Run `{' '.join(argv)}` — {ALLOWED_COMMANDS[args['name']][1]}")


@gateway.operation("shell.run_named", _risk, "Run an allow-listed named command", preview=_preview)
async def run_named(name: str, params: Optional[Dict[str, str]] = None, timeout: float = 120.0) -> OpResult:
    argv, err = _build_argv(name, dict(params or {}))
    if argv is None:
        return OpResult.fail(err)

    def _run() -> subprocess.CompletedProcess:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, shell=False)

    try:
        proc = await asyncio.to_thread(_run)
    except FileNotFoundError:
        return OpResult.fail(f"'{argv[0]}' is not installed or not on PATH.")
    except subprocess.TimeoutExpired:
        return OpResult.fail(f"'{name}' timed out after {timeout:.0f}s.")
    out = (proc.stdout or "")[-6000:]
    errout = (proc.stderr or "")[-2000:]
    data = {"exit_code": proc.returncode, "stdout": out, "stderr": errout, "argv": argv}
    summary = out.strip() or errout.strip() or "(no output)"
    return OpResult(proc.returncode == 0, summary[-3000:], data=data)
