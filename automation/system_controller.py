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


import re


def resolve_user_system_path(path_str: str) -> Path:
    """
    Resolve user-friendly path names like 'desktop', 'downloads', 'documents',
    or expressions like 'size of folder named ola in desktop' or 'ola in desktop'.
    """
    raw = path_str.strip().strip("'\"")
    location_hint = ""

    # Check for expressions like "folder named ola" or "file named 123"
    m_folder = re.search(
        r"(?:folder|directory|file)\s+(?:named\s+|called\s+)?['\"]?([a-zA-Z0-9_\-\./\\]+)['\"]?",
        raw,
        re.IGNORECASE,
    )
    if m_folder:
        clean_target = m_folder.group(1).strip()
    else:
        clean_target = raw

    # Clean location prepositions
    loc_match = re.search(
        r"(?:in|on|at|inside)\s+(?:my\s+)?(desktop|downloads|documents|docs|pictures|music|videos)(?:\s+folder|\s+directory)?",
        raw,
        re.IGNORECASE,
    )
    if loc_match:
        location_hint = loc_match.group(1).lower().strip()

    # Check for prefixes like "desktop/123" or "desktop\123"
    if clean_target.lower().startswith("desktop/") or clean_target.lower().startswith("desktop\\"):
        location_hint = "desktop"
        clean_target = clean_target[8:].strip()
    elif clean_target.lower().startswith("downloads/") or clean_target.lower().startswith("downloads\\"):
        location_hint = "downloads"
        clean_target = clean_target[10:].strip()
    elif clean_target.lower().startswith("documents/") or clean_target.lower().startswith("documents\\"):
        location_hint = "documents"
        clean_target = clean_target[10:].strip()

    user_home = Path.home()
    onedrive_desktop = user_home / "OneDrive" / "Desktop"
    onedrive_docs = user_home / "OneDrive" / "Documents"
    onedrive_pics = user_home / "OneDrive" / "Pictures"

    target_dir = Path.cwd()
    if location_hint == "desktop" or "desktop" in raw.lower():
        target_dir = onedrive_desktop if onedrive_desktop.exists() else (user_home / "Desktop")
    elif location_hint in ("downloads", "download"):
        target_dir = user_home / "Downloads"
    elif location_hint in ("documents", "docs"):
        target_dir = onedrive_docs if onedrive_docs.exists() else (user_home / "Documents")
    elif location_hint in ("pictures", "pics"):
        target_dir = onedrive_pics if onedrive_pics.exists() else (user_home / "Pictures")
    elif location_hint == "music":
        target_dir = user_home / "Music"
    elif location_hint == "videos":
        target_dir = user_home / "Videos"

    # If the target already exists as a folder or file directly
    cand = target_dir / clean_target
    if cand.exists():
        return cand

    # Check with .txt extension if a file was expected
    cand_txt = target_dir / f"{clean_target}.txt"
    if cand_txt.exists():
        return cand_txt

    if location_hint:
        return cand

    p = Path(raw)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


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
            p = resolve_user_system_path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            log_action("system", "create_file", str(p), result="CREATED")
            return {"success": True, "action": "create_file", "path": str(p), "size": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_path_size(self, path: str) -> dict:
        """Calculate total size of a file or directory in bytes, KB, MB, GB."""
        log_action("system", "get_path_size", path, dry_run=self.dry_run)
        try:
            p = resolve_user_system_path(path)
            if not p.exists():
                return {"success": False, "error": f"Path not found: '{path}' (checked '{p}')"}

            if p.is_file():
                size_bytes = p.stat().st_size
                num_files = 1
            else:
                size_bytes = 0
                num_files = 0
                for root, _, files in os.walk(p):
                    for f in files:
                        try:
                            fp = os.path.join(root, f)
                            size_bytes += os.path.getsize(fp)
                            num_files += 1
                        except Exception:
                            pass

            def format_bytes(b: int) -> str:
                if b < 1024:
                    return f"{b} B"
                elif b < 1024 * 1024:
                    return f"{b / 1024:.2f} KB"
                elif b < 1024 * 1024 * 1024:
                    return f"{b / (1024 * 1024):.2f} MB"
                else:
                    return f"{b / (1024 * 1024 * 1024):.2f} GB"

            return {
                "success": True,
                "path": str(p),
                "name": p.name,
                "is_dir": p.is_dir(),
                "size_bytes": size_bytes,
                "formatted_size": format_bytes(size_bytes),
                "num_files": num_files,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def list_directory(self, path: str = "") -> dict:
        """List files and folders in the target directory."""
        log_action("system", "list_directory", path, dry_run=self.dry_run)
        try:
            p = resolve_user_system_path(path) if path else Path.cwd()
            if not p.exists() or not p.is_dir():
                return {"success": False, "error": f"Directory not found: '{path}' (checked '{p}')"}

            items = []
            for entry in p.iterdir():
                try:
                    items.append({
                        "name": entry.name,
                        "is_dir": entry.is_dir(),
                        "size_bytes": entry.stat().st_size if entry.is_file() else 0,
                    })
                except Exception:
                    pass
            return {"success": True, "path": str(p), "items": items, "total_items": len(items)}
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

