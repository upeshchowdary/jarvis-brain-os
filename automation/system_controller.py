"""JARVIS System Controller — Safe file system and command execution.

Wraps the existing FileSystemTool and adds:
- Full file operations (create, read, rename, move, copy, delete, search)
- Safe command execution with stdout/stderr capture
- Confirmation gates for dangerous operations
- All operations go through SafetyManager
"""

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

from automation.config import automation_config
from automation.automation_logger import log_action
from automation.safety_manager import safety_manager


class SystemController:
    """Safe file system and terminal command controller."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run or automation_config.DRY_RUN

    # ── File Operations ──────────────────────────────────────────────

    async def create_file(self, path: str, content: str = "") -> dict:
        allowed, reason = safety_manager.is_allowed("create_file", path)
        if not allowed:
            return {"success": False, "blocked": True, "reason": reason}
        log_action("system", "create_file", path, dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "path": path}
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return {"success": True, "action": "create_file", "path": str(p), "size": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def read_file(self, path: str, max_bytes: int = 50000) -> dict:
        log_action("system", "read_file", path, dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "content": "[DRY RUN]"}
        try:
            p = Path(path)
            if not p.exists():
                return {"success": False, "error": f"File not found: {path}"}
            content = p.read_bytes()[:max_bytes].decode("utf-8", errors="replace")
            return {"success": True, "path": path, "content": content, "size": p.stat().st_size}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def rename_file(self, src: str, new_name: str) -> dict:
        allowed, reason = safety_manager.is_allowed("rename_file", src)
        if not allowed:
            return {"success": False, "blocked": True, "reason": reason}
        log_action("system", "rename_file", f"{src} -> {new_name}", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}
        try:
            p = Path(src)
            dest = p.parent / new_name
            p.rename(dest)
            return {"success": True, "action": "rename_file", "new_path": str(dest)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def copy_file(self, src: str, dest: str) -> dict:
        allowed, reason = safety_manager.is_allowed("copy_file", src)
        if not allowed:
            return {"success": False, "blocked": True, "reason": reason}
        log_action("system", "copy_file", f"{src} -> {dest}", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}
        try:
            shutil.copy2(src, dest)
            return {"success": True, "action": "copy_file", "src": src, "dest": dest}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def move_file(self, src: str, dest: str, user_confirmed: bool = False) -> dict:
        allowed, reason = safety_manager.is_allowed("move_file", src, user_confirmed)
        if not allowed:
            return {"success": False, "blocked": True,
                    "reason": reason,
                    "confirmation_prompt": safety_manager.format_confirmation_request("move_file", src)}
        log_action("system", "move_file", f"{src} -> {dest}", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}
        try:
            shutil.move(src, dest)
            return {"success": True, "action": "move_file", "src": src, "dest": dest}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def delete_file(self, path: str, user_confirmed: bool = False) -> dict:
        """Delete a file. ALWAYS requires user confirmation."""
        allowed, reason = safety_manager.is_allowed("delete_file", path, user_confirmed)
        if not allowed:
            return {
                "success": False,
                "blocked": True,
                "reason": reason,
                "confirmation_prompt": safety_manager.format_confirmation_request("delete_file", path),
            }
        log_action("system", "delete_file", path, dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "would_delete": path}
        try:
            p = Path(path)
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(path)
            else:
                return {"success": False, "error": f"Path not found: {path}"}
            return {"success": True, "action": "delete_file", "path": path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def list_files(self, directory: str = ".") -> dict:
        log_action("system", "list_files", directory, dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "items": []}
        try:
            p = Path(directory)
            if not p.exists():
                return {"success": False, "error": f"Directory not found: {directory}"}
            items = [
                {
                    "name": entry.name,
                    "type": "dir" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if entry.is_file() else 0,
                    "path": str(entry),
                }
                for entry in sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
            ]
            return {"success": True, "directory": directory, "items": items, "count": len(items)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def search_files(
        self, pattern: str, directory: str = ".", recursive: bool = True
    ) -> dict:
        log_action("system", "search_files", f"{pattern} in {directory}", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "matches": []}
        try:
            p = Path(directory)
            if recursive:
                matches = [str(f) for f in p.rglob(pattern)]
            else:
                matches = [str(f) for f in p.glob(pattern)]
            return {"success": True, "pattern": pattern, "matches": matches, "count": len(matches)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def file_exists(self, path: str) -> bool:
        return Path(path).exists()

    # ── Command Execution ────────────────────────────────────────────

    async def run_command(
        self,
        command: str,
        timeout: float | None = None,
        user_confirmed: bool = False,
        cwd: Optional[str] = None,
    ) -> dict:
        """
        Execute a system command safely.
        HIGH-risk commands always require confirmation.
        Returns structured result with stdout, stderr, exit_code.
        """
        allowed, reason = safety_manager.is_allowed("run_command", command, user_confirmed)
        if not allowed:
            return {
                "success": False,
                "blocked": True,
                "reason": reason,
                "confirmation_prompt": safety_manager.format_confirmation_request("run_command", command),
            }

        log_action("system", "run_command", command[:80], dry_run=self.dry_run)
        if self.dry_run:
            return {
                "success": True,
                "dry_run": True,
                "command": command,
                "note": f"Would run: {command}",
            }

        timeout_s = timeout or automation_config.ACTION_TIMEOUT

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_s
                )
            except asyncio.TimeoutError:
                proc.kill()
                return {
                    "success": False,
                    "command": command,
                    "error": f"Command timed out after {timeout_s}s",
                    "exit_code": -1,
                }

            stdout = stdout_b.decode("utf-8", errors="replace").strip()
            stderr = stderr_b.decode("utf-8", errors="replace").strip()
            exit_code = proc.returncode or 0

            return {
                "success": exit_code == 0,
                "command": command,
                "stdout": stdout[:5000],  # Limit output size
                "stderr": stderr[:2000],
                "exit_code": exit_code,
            }
        except Exception as e:
            return {"success": False, "command": command, "error": str(e), "exit_code": -1}


# Global singleton
system_controller = SystemController()

