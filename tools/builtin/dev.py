"""Developer helper tools: find projects, git status / today's commits, run tests, open in VS Code."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, List, Optional

from automation.gateway import ActionRequest, Risk
from core.paths import data_path
from tools.base import Tool, ToolContext, ToolResult, obj, via_gateway
from tools.registry import register

SEARCH_ROOTS = ["~/projects", "~/Projects", "~/code", "~/Code", "~/dev", "~/source/repos", "~/repos", "~/Desktop",
                "~/Documents", "~/OneDrive/Desktop", "~/OneDrive/Documents", "~/PycharmProjects", "~/IdeaProjects", "~"]


def find_project(name: str) -> Optional[Path]:
    """Locate a project folder by (fuzzy) name. Remembers matches in data/projects.json."""
    key = name.strip().lower().replace(" project", "").replace("my ", "").strip()
    p = Path(name).expanduser()
    if p.is_dir():
        return p
    mem_file = data_path("projects.json")
    try:
        known = json.loads(mem_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        known = {}
    if key in known and Path(known[key]).is_dir():
        return Path(known[key])
    want = key.replace(" ", "").replace("-", "").replace("_", "")
    best: Optional[Path] = None
    for root in SEARCH_ROOTS:
        base = Path(root).expanduser()
        if not base.is_dir():
            continue
        try:
            for child in base.iterdir():
                if not child.is_dir() or child.name.startswith("."):
                    continue
                norm = child.name.lower().replace("-", "").replace("_", "").replace(" ", "")
                if norm == want or (want in norm and (child / ".git").exists()):
                    best = child
                    break
        except OSError:
            continue
        if best:
            break
    if best:
        known[key] = str(best)
        mem_file.write_text(json.dumps(known, indent=2), encoding="utf-8")
    return best


class ProjectTool(Tool):
    async def _project(self, project: str) -> Optional[Path]:
        return await asyncio.to_thread(find_project, project or ".")


@register
class GitStatus(ProjectTool):
    name = "git_status"
    description = "Show git status and uncommitted changes of a project."
    parameters = obj({"project": {"type": "string", "description": "Project name or path"}}, ["project"])
    side_effect = True  # runs a (read-only, allow-listed) command through the gateway

    async def run(self, ctx: ToolContext, project: str = "", **_: Any) -> ToolResult:
        path = await self._project(project)
        if path is None:
            return ToolResult.error(f"I couldn't find a project called “{project}”. Tell me its folder path once and I'll remember it.")
        return await via_gateway(ctx, ActionRequest("shell.run_named", {"name": "git_status", "params": {"path": str(path)}}))


@register
class GitToday(ProjectTool):
    name = "git_today_commits"
    description = "List today's commits in a project so you can summarise them."
    parameters = obj({"project": {"type": "string"}}, ["project"])
    side_effect = True
    final = False

    async def run(self, ctx: ToolContext, project: str = "", **_: Any) -> ToolResult:
        path = await self._project(project)
        if path is None:
            return ToolResult.error(f"I couldn't find a project called “{project}”.")
        res = await via_gateway(ctx, ActionRequest("shell.run_named", {"name": "git_log_today", "params": {"path": str(path)}}))
        if res.success and res.status == "ok":
            res.content_for_llm = res.message
            res.message = f"Today's commits in {path.name}:"
        return res


@register
class RunTests(ProjectTool):
    name = "run_tests"
    description = "Run a project's test suite (pytest or npm test). Asks for confirmation because it executes project code."
    parameters = obj({"project": {"type": "string"}}, ["project"])
    side_effect = True
    risk = Risk.HIGH
    final = False

    async def run(self, ctx: ToolContext, project: str = "", **_: Any) -> ToolResult:
        path = await self._project(project)
        if path is None:
            return ToolResult.error(f"I couldn't find a project called “{project}”.")
        runner = "npm_test" if (path / "package.json").exists() and not (path / "pyproject.toml").exists() else "pytest"
        res = await via_gateway(ctx, ActionRequest("shell.run_named", {"name": runner, "params": {"path": str(path)}}))
        if res.status == "ok" or res.status == "error":
            res.content_for_llm = res.message
        return res


@register
class OpenProject(ProjectTool):
    name = "open_project"
    description = "Open a project folder in VS Code."
    parameters = obj({"project": {"type": "string"}}, ["project"])
    side_effect = True

    async def run(self, ctx: ToolContext, project: str = "", **_: Any) -> ToolResult:
        path = await self._project(project)
        if path is None:
            return ToolResult.error(f"I couldn't find a project called “{project}”.")
        return await via_gateway(ctx, ActionRequest("dev.open_in_editor", {"path": str(path)}))


__all__: List[str] = ["find_project"]
