"""Abstract Interface for Tools in JARVIS AI Operating System."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    success: bool
    data: Optional[Any] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseTool(ABC):
    """Abstract Base Class for all future tools (Browser, Filesystem, Terminal, Vision, etc.)."""

    name: str
    description: str
    version: str = "1.0.0"

    @abstractmethod
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return OpenAPI JSON Schema of required and optional parameters for tool invocation."""
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Asynchronously execute the tool logic."""
        pass
