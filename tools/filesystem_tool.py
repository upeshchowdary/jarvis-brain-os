"""Workspace File System Tool for reading, writing, and listing project files safely."""

import os
from pathlib import Path
from typing import Dict, Any, List
from brain.tool_router import BaseBrainTool
from brain.brain_config import brain_config
from brain.logger import logger


class FileSystemTool(BaseBrainTool):
    """Tool for reading, writing, listing, and inspecting workspace files."""

    name: str = "filesystem"
    description: str = "Read, write, list, or inspect local files within the workspace project directory."
    version: str = "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "write", "list"],
                    "description": "File action to perform: 'read', 'write', or 'list'.",
                },
                "path": {
                    "type": "string",
                    "description": "Relative file or directory path within the workspace.",
                },
                "content": {
                    "type": "string",
                    "description": "Text content to write if action is 'write'.",
                },
            },
            "required": ["action", "path"],
        }

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        action = str(kwargs.get("action", "")).lower()
        rel_path = str(kwargs.get("path", "")).strip()
        content = kwargs.get("content", "")

        if not rel_path:
            return {"success": False, "error": "Path parameter is required."}

        # Security sandbox check: path must remain within project base directory
        base_dir = brain_config.base_dir.resolve()
        target_path = (base_dir / rel_path).resolve()

        if not str(target_path).startswith(str(base_dir)):
            return {"success": False, "error": "Access denied: Path is outside the project workspace sandbox."}

        logger.info(f"FileSystemTool executing action '{action}' on path: '{rel_path}'")

        try:
            if action == "read":
                if not target_path.exists() or not target_path.is_file():
                    return {"success": False, "error": f"File '{rel_path}' does not exist."}
                file_text = target_path.read_text(encoding="utf-8")
                return {"success": True, "path": rel_path, "content": file_text, "size_bytes": len(file_text)}

            elif action == "write":
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(content, encoding="utf-8")
                return {"success": True, "path": rel_path, "bytes_written": len(content)}

            elif action == "list":
                if not target_path.exists() or not target_path.is_dir():
                    return {"success": False, "error": f"Directory '{rel_path}' does not exist."}
                items = []
                for entry in target_path.iterdir():
                    items.append({
                        "name": entry.name,
                        "is_dir": entry.is_dir(),
                        "size_bytes": entry.stat().st_size if entry.is_file() else 0,
                    })
                return {"success": True, "path": rel_path, "items": items}

            else:
                return {"success": False, "error": f"Unsupported action '{action}'. Valid actions: read, write, list."}

        except Exception as exc:
            logger.error(f"FileSystemTool execution error: {exc}")
            return {"success": False, "error": str(exc)}
