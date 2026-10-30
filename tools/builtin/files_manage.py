"""File-management tools: create/move/rename/delete (Recycle Bin), organize, bulk rename, zip,
duplicates, content search, undo. Writes go through the gateway; searches are read-only."""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from automation.gateway import ActionRequest, Risk, gateway
from tools.base import Tool, ToolContext, ToolResult, from_gateway, obj, via_gateway
from tools.registry import register

_LOC = {"type": "string", "description": "Path or natural description, e.g. 'downloads', 'report.pdf on my desktop'"}


def _resolve(loc: str) -> Path:
    from automation.paths import resolve

    return resolve(loc)


class FileWrite(Tool):
    side_effect = True
    risk = Risk.MEDIUM


@register
class CreateFile(FileWrite):
    name = "file_create"
    description = "Create a text file with the given content."
    parameters = obj({"path": _LOC, "content": {"type": "string"}, "overwrite": {"type": "boolean"}}, ["path"])

    async def run(self, ctx: ToolContext, path: str = "", content: str = "", overwrite: bool = False, **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("file.create", {"path": path, "content": content, "overwrite": overwrite}))


@register
class AppendFile(FileWrite):
    name = "file_append"
    description = "Append text to a file."
    parameters = obj({"path": _LOC, "content": {"type": "string"}}, ["path", "content"])

    async def run(self, ctx: ToolContext, path: str = "", content: str = "", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("file.append", {"path": path, "content": content}))


@register
class MakeFolder(FileWrite):
    name = "folder_create"
    description = "Create a folder."
    parameters = obj({"path": _LOC}, ["path"])

    async def run(self, ctx: ToolContext, path: str = "", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("file.mkdir", {"path": path}))


@register
class RenameFile(FileWrite):
    name = "file_rename"
    description = "Rename a file or folder."
    parameters = obj({"src": _LOC, "new_name": {"type": "string"}}, ["src", "new_name"])

    async def run(self, ctx: ToolContext, src: str = "", new_name: str = "", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("file.rename", {"src": src, "new_name": new_name}))


@register
class MoveFile(FileWrite):
    name = "file_move"
    description = "Move a file/folder to another folder."
    parameters = obj({"src": _LOC, "dest": _LOC}, ["src", "dest"])
    risk = Risk.HIGH

    async def run(self, ctx: ToolContext, src: str = "", dest: str = "", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("file.move", {"src": src, "dest": dest}))


@register
class CopyFile(FileWrite):
    name = "file_copy"
    description = "Copy a file/folder."
    parameters = obj({"src": _LOC, "dest": _LOC}, ["src", "dest"])

    async def run(self, ctx: ToolContext, src: str = "", dest: str = "", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("file.copy", {"src": src, "dest": dest}))


@register
class DeleteFiles(FileWrite):
    name = "file_delete"
    description = "Delete files/folders by moving them to the Recycle Bin (never permanent). Always confirmed."
    parameters = obj({"paths": {"type": "array", "items": {"type": "string"}}}, ["paths"])
    risk = Risk.HIGH

    async def run(self, ctx: ToolContext, paths: Any = None, **_: Any) -> ToolResult:
        paths = [paths] if isinstance(paths, str) else list(paths or [])
        return await via_gateway(ctx, ActionRequest("file.trash", {"paths": paths}))


@register
class OrganizeFolder(FileWrite):
    name = "organize_folder"
    description = "Tidy a folder (e.g. Downloads) by moving files into subfolders by type (Images, Documents…) or by month. Undoable."
    parameters = obj({"folder": _LOC, "by": {"type": "string", "enum": ["type", "date"]}}, ["folder"])
    risk = Risk.HIGH

    async def run(self, ctx: ToolContext, folder: str = "downloads", by: str = "type", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("files.organize", {"folder": folder, "by": by or "type"}))


@register
class BulkRename(FileWrite):
    name = "bulk_rename"
    description = "Rename many files: find/replace text in names, or a template like 'trip_{n}' ({n}=number, {name}=old name). Shows a preview first."
    parameters = obj({"folder": _LOC, "find": {"type": "string"}, "replace": {"type": "string"},
                      "template": {"type": "string"}, "pattern": {"type": "string", "description": "glob, e.g. *.jpg"}}, ["folder"])

    async def run(self, ctx: ToolContext, folder: str = "", find: str = "", replace: str = "", template: str = "", pattern: str = "*", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("files.bulk_rename", {"folder": folder, "find": find, "replace": replace,
                                                                          "template": template, "pattern": pattern or "*"}))


@register
class ZipFiles(FileWrite):
    name = "zip_files"
    description = "Compress files/folders into a .zip."
    parameters = obj({"paths": {"type": "array", "items": {"type": "string"}}, "dest": _LOC}, ["paths", "dest"])

    async def run(self, ctx: ToolContext, paths: Any = None, dest: str = "", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("files.zip", {"paths": list(paths or []), "dest": dest}))


@register
class UnzipFile(FileWrite):
    name = "unzip_file"
    description = "Extract a .zip archive."
    parameters = obj({"archive": _LOC, "dest": _LOC}, ["archive"])

    async def run(self, ctx: ToolContext, archive: str = "", dest: str = "", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("files.unzip", {"archive": archive, "dest": dest}))


@register
class UndoLast(Tool):
    name = "undo_last_action"
    description = "Undo the most recent file operation (move, rename, organize, create, bulk rename)."
    parameters = obj({})
    side_effect = True
    risk = Risk.MEDIUM

    async def run(self, ctx: ToolContext, **_: Any) -> ToolResult:
        return from_gateway(await gateway.undo_last(ctx.session_id))


# ── Read-only file intelligence ───────────────────────────────────────────

def _walk(root: Path, limit: int = 20000):
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("node_modules", "__pycache__", ".git", "AppData")]
        for f in filenames:
            yield Path(dirpath) / f
            count += 1
            if count >= limit:
                return


def find_duplicates(root: Path) -> List[List[str]]:
    by_size: Dict[int, List[Path]] = defaultdict(list)
    for f in _walk(root):
        try:
            size = f.stat().st_size
        except OSError:
            continue
        if size > 0:
            by_size[size].append(f)
    groups: List[List[str]] = []
    for size, files in by_size.items():
        if len(files) < 2:
            continue
        by_hash: Dict[str, List[str]] = defaultdict(list)
        for f in files:
            h = hashlib.sha256()
            try:
                with open(f, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
            except OSError:
                continue
            by_hash[h.hexdigest()].append(str(f))
        groups += [g for g in by_hash.values() if len(g) > 1]
    return sorted(groups, key=lambda g: -os.path.getsize(g[0]))


@register
class FindDuplicates(Tool):
    name = "find_duplicate_files"
    description = "Find identical files (by content hash) in a folder and report wasted space."
    parameters = obj({"folder": _LOC}, ["folder"])
    sensitive_output = True

    async def run(self, ctx: ToolContext, folder: str = "downloads", **_: Any) -> ToolResult:
        root = _resolve(folder)
        if not root.is_dir():
            return ToolResult.error(f"{root} isn't a folder.")
        groups = await asyncio.to_thread(find_duplicates, root)
        if not groups:
            return ToolResult.ok(f"No duplicate files in {root}.")
        wasted = sum(os.path.getsize(g[0]) * (len(g) - 1) for g in groups)
        lines = [f"• {len(g)}× {Path(g[0]).name} ({os.path.getsize(g[0]) // 1024} KB)" for g in groups[:15]]
        return ToolResult.ok(f"Found {len(groups)} sets of duplicates wasting {wasted / 1048576:.1f} MB:\n" + "\n".join(lines)
                             + "\nSay “delete the duplicate copies” to move the extras to the Recycle Bin.", data=groups)


@register
class FindFiles(Tool):
    name = "find_files"
    description = "Find files by name/glob and/or modified date (e.g. pattern '*.pdf', modified 'today'|'this week')."
    parameters = obj({"folder": _LOC, "pattern": {"type": "string"}, "modified": {"type": "string", "enum": ["", "today", "this week", "this month"]}}, ["folder"])
    sensitive_output = True

    async def run(self, ctx: ToolContext, folder: str = "downloads", pattern: str = "*", modified: str = "", **_: Any) -> ToolResult:
        root = _resolve(folder)
        cutoff = {"today": 1, "this week": 7, "this month": 31}.get(modified)
        since = datetime.now().timestamp() - cutoff * 86400 if cutoff else 0
        if modified == "today":
            since = datetime.now().replace(hour=0, minute=0, second=0).timestamp()

        def _find():
            hits = []
            for f in _walk(root):
                if fnmatch.fnmatch(f.name.lower(), (pattern or "*").lower()):
                    try:
                        mt = f.stat().st_mtime
                    except OSError:
                        continue
                    if mt >= since:
                        hits.append((mt, f))
            return sorted(hits, reverse=True)[:30]

        hits = await asyncio.to_thread(_find)
        if not hits:
            return ToolResult.ok("No matching files found.")
        lines = [f"• {f.name} — {datetime.fromtimestamp(mt).strftime('%d %b %H:%M')} ({f.parent})" for mt, f in hits]
        return ToolResult.ok(f"Found {len(hits)} file(s):\n" + "\n".join(lines), data=[str(f) for _, f in hits])


@register
class SearchContents(Tool):
    name = "search_file_contents"
    description = "Search inside text/code files in a folder for a word or phrase."
    parameters = obj({"folder": _LOC, "text": {"type": "string"}}, ["folder", "text"])
    sensitive_output = True
    untrusted_output = True
    final = False

    async def run(self, ctx: ToolContext, folder: str = "", text: str = "", **_: Any) -> ToolResult:
        root = _resolve(folder)
        needle = text.lower()
        exts = {".txt", ".md", ".py", ".js", ".ts", ".json", ".csv", ".html", ".css", ".java", ".c", ".cpp", ".yaml", ".yml", ".log", ".ini"}

        def _search():
            hits = []
            for f in _walk(root, 5000):
                if f.suffix.lower() not in exts:
                    continue
                try:
                    with open(f, encoding="utf-8", errors="ignore") as fh:
                        for n, line in enumerate(fh, 1):
                            if needle in line.lower():
                                hits.append(f"{f}:{n}: {line.strip()[:160]}")
                                break
                except OSError:
                    continue
                if len(hits) >= 40:
                    break
            return hits

        hits = await asyncio.to_thread(_search)
        if not hits:
            return ToolResult.ok(f"“{text}” wasn't found in {root}.")
        return ToolResult.ok(f"Found “{text}” in {len(hits)} file(s).", data=hits, content_for_llm="\n".join(hits))
