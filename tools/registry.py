"""Central tool registry."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from core.logging import logger
from tools.base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if not tool.name:
            raise ValueError("Tool must have a name")
        if tool.side_effect is False and tool.risk.label != "low":
            raise ValueError(f"Read-only tool {tool.name} must be LOW risk")
        self._tools[tool.name] = tool
        return tool

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def all(self) -> List[Tool]:
        return list(self._tools.values())

    def read_only(self) -> List[Tool]:
        return [t for t in self._tools.values() if not t.side_effect]

    def select(self, names: Iterable[str]) -> List[Tool]:
        return [self._tools[n] for n in names if n in self._tools]

    def info(self) -> List[dict]:
        return [t.info() for t in sorted(self._tools.values(), key=lambda t: t.name)]


registry = ToolRegistry()


def register(tool_cls):
    """Class decorator: instantiate and register a Tool subclass."""
    registry.register(tool_cls())
    return tool_cls


_loaded = False


def load_builtin_tools() -> ToolRegistry:
    global _loaded
    if not _loaded:
        _loaded = True
        import tools.builtin  # noqa: F401  (registers tools)
        logger.debug(f"[Tools] {len(registry.all())} tools registered")
    return registry
