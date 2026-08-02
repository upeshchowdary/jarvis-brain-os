"""Abstract Interface for LLM Providers."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from app.domain.models.chat import ChatMessage
from app.domain.models.response import LLMResponseMetadata


class LLMResult(BaseModel):
    """Result container returned by LLM Provider execution."""
    content: str
    metadata: LLMResponseMetadata
    raw_response: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """Abstract Base Class that all concrete LLM provider adapters must inherit from."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, **kwargs: Any) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.extra_kwargs = kwargs

    @abstractmethod
    async def generate_completion(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResult:
        """Asynchronously generate a completion response from the provider."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify provider availability and authentication status."""
        pass
