"""Tool Registry for dynamic tool discovery, registration, and invocation."""

from typing import Dict, List, Optional, Any
from app.domain.interfaces.tool import BaseTool, ToolResult
from app.utils.exceptions import ToolExecutionError
from app.utils.logger import logger


class ToolRegistry:
    """Central registry where tools can be dynamically registered and dispatched."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a new tool instance in the registry."""
        if tool.name in self._tools:
            logger.warning(f"Overwriting previously registered tool: '{tool.name}'")
        self._tools[tool.name] = tool
        logger.info(f"Registered tool '{tool.name}' (v{tool.version})")

    def unregister(self, tool_name: str) -> bool:
        """Remove a tool from the registry."""
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info(f"Unregistered tool '{tool_name}'")
            return True
        return False

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Retrieve tool by name."""
        return self._tools.get(tool_name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return list of schemas for all currently registered tools."""
        result = []
        for tool in self._tools.values():
            result.append({
                "name": tool.name,
                "description": tool.description,
                "version": tool.version,
                "parameters_schema": tool.get_parameters_schema(),
            })
        return result

    async def execute_tool(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name with arguments."""
        tool = self.get_tool(tool_name)
        if not tool:
            logger.error(f"Attempted to execute unregistered tool: '{tool_name}'")
            raise ToolExecutionError(f"Tool '{tool_name}' is not registered in JARVIS Operating System.")

        try:
            logger.info(f"Executing tool '{tool_name}' with kwargs: {kwargs}")
            return await tool.execute(**kwargs)
        except Exception as exc:
            logger.error(f"Error executing tool '{tool_name}': {exc}")
            raise ToolExecutionError(f"Failed to execute tool '{tool_name}': {exc}") from exc


# Global tool registry singleton
tool_registry = ToolRegistry()
