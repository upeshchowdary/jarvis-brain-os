"""Developer helper operations."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

from automation.gateway import OpResult, Risk, gateway


@gateway.operation("dev.open_in_editor", Risk.LOW, "Open a folder/file in VS Code")
async def open_in_editor(path: str) -> OpResult:
    p = Path(path).expanduser()
    if not p.exists():
        return OpResult.fail(f"{p} doesn't exist.")
    exe = shutil.which("code") or shutil.which("code.cmd")
    if not exe:
        return OpResult.fail("VS Code's 'code' command isn't on PATH (VS Code → Command Palette → 'Shell Command: Install code in PATH').")
    await asyncio.to_thread(lambda: subprocess.Popen([exe, str(p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    return OpResult.ok(f"Opened {p.name} in VS Code.")
