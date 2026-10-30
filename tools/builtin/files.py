"""Read-only file tools. File contents are untrusted and sensitive (PRIVACY_MODE aware)."""

from __future__ import annotations

from typing import Any

from automation.files_read import file_reader
from tools.base import Tool, ToolContext, ToolResult, obj
from tools.registry import register

_LOCATION = {"type": "string", "description": "A path or natural description, e.g. 'desktop', 'downloads folder', 'notes.txt on my desktop'"}


@register
class ListFiles(Tool):
    name = "files_list"
    description = "List (or count) the files and folders in a directory such as Desktop, Downloads, Documents."
    parameters = obj({"location": _LOCATION}, ["location"])
    sensitive_output = True

    async def run(self, ctx: ToolContext, location: str = "desktop", **_: Any) -> ToolResult:
        res = await file_reader.list_directory(location)
        if not res.get("success"):
            return ToolResult.error(res.get("error", "Couldn't list that folder."))
        items = sorted(res.get("items", []), key=lambda i: (not i.get("is_dir"), i.get("name", "").lower()))
        files = sum(1 for i in items if not i.get("is_dir"))
        lines = [f"  {'📁' if i.get('is_dir') else '📄'} {i.get('name')}" for i in items[:50]]
        more = f"\n  … and {len(items) - 50} more" if len(items) > 50 else ""
        msg = f"`{res['path']}` has {len(items)} items ({files} files, {len(items) - files} folders):\n" + ("\n".join(lines) or "  (empty)") + more
        return ToolResult.ok(msg, data=res)


@register
class FileSize(Tool):
    name = "files_size"
    description = "Get the size of a file or folder."
    parameters = obj({"location": _LOCATION}, ["location"])
    sensitive_output = True

    async def run(self, ctx: ToolContext, location: str = "", **_: Any) -> ToolResult:
        res = await file_reader.get_path_size(location)
        if not res.get("success"):
            return ToolResult.error(res.get("error", "Not found."))
        kind = "folder" if res.get("is_dir") else "file"
        extra = f" across {res['num_files']} files" if res.get("is_dir") else ""
        return ToolResult.ok(f"The {kind} “{res['name']}” is {res['formatted_size']}{extra} (`{res['path']}`).", data=res)


@register
class FileInfo(Tool):
    name = "files_info"
    description = "Check whether a file/folder exists and show its type, size and location."
    parameters = obj({"location": _LOCATION}, ["location"])
    sensitive_output = True

    async def run(self, ctx: ToolContext, location: str = "", **_: Any) -> ToolResult:
        res = await file_reader.inspect_file(location)
        if not res.get("exists"):
            return ToolResult.ok(f"I couldn't find that: {res.get('error', 'it does not exist')}.", data=res)
        kind = "folder" if res.get("is_dir") else "file"
        return ToolResult.ok(
            f"Yes — the {kind} `{res.get('name')}` exists at `{res.get('path')}` ({res.get('file_type', '')}, {res.get('formatted_size', '')}).",
            data={k: v for k, v in res.items() if k != "content_preview"},
        )


@register
class ReadFile(Tool):
    name = "files_read"
    description = "Read a text/PDF/DOCX file so you can summarise it or answer a question about it."
    parameters = obj({"location": _LOCATION}, ["location"])
    untrusted_output = True
    sensitive_output = True
    final = False

    async def run(self, ctx: ToolContext, location: str = "", **_: Any) -> ToolResult:
        res = await file_reader.read_file(location, max_chars=12000)
        if not res.get("success"):
            return ToolResult.error(res.get("error", "Couldn't read that file."))
        content = res.get("content", "")
        meta = f"{res.get('name')} ({res.get('size_bytes', 0)} bytes, {res.get('line_count', 0)} lines) at {res.get('path')}"
        return ToolResult.ok(f"Read {meta}.", data={k: v for k, v in res.items() if k != "content"}, content_for_llm=f"File: {meta}\n\n{content}")
