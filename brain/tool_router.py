"""Tool Router Infrastructure for tool registration, tool discovery, and JSON tool calling."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from brain.logger import logger


class ToolCallSpec(BaseModel):
    tool: str = Field(..., description="Target registered tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="JSON arguments for tool invocation")


class BaseBrainTool(ABC):
    """Abstract protocol for future brain tools."""

    name: str
    description: str
    version: str = "1.0.0"

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Return parameter schema for tool invocation."""
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute tool logic asynchronously."""
        pass


class ToolRouter:
    """Central registry and router for tool discovery and JSON dispatching."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseBrainTool] = {}

    def register_tool(self, tool: BaseBrainTool) -> None:
        """Register a new tool instance in the router."""
        if tool.name in self._tools:
            logger.warning(f"Overwriting registered tool: '{tool.name}'")
        self._tools[tool.name] = tool
        logger.info(f"ToolRouter registered tool '{tool.name}' (v{tool.version})")

    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool by name."""
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info(f"ToolRouter unregistered tool '{tool_name}'")
            return True
        return False

    def discover_tools(self) -> List[Dict[str, Any]]:
        """Return list of JSON schemas for all registered tools."""
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "version": tool.version,
                "parameters": tool.get_schema(),
            })
        return schemas

    async def route_and_execute(self, tool_call: ToolCallSpec) -> Dict[str, Any]:
        """Dispatch JSON tool call to the corresponding registered tool."""
        tool_name = tool_call.tool
        if tool_name not in self._tools:
            logger.error(f"Attempted to execute unregistered tool: '{tool_name}'")
            return {
                "success": False,
                "error": f"Tool '{tool_name}' is not registered in ToolRouter.",
            }

        tool = self._tools[tool_name]
        try:
            logger.info(f"ToolRouter executing tool '{tool_name}' with args: {tool_call.arguments}")
            result = await tool.execute(**tool_call.arguments)
            return {"success": True, "data": result}
        except Exception as exc:
            logger.error(f"Error executing tool '{tool_name}': {exc}")
            return {"success": False, "error": str(exc)}


# Global tool router singleton
tool_router = ToolRouter()
