"""Base LLM Provider Implementation with Unified HTTP handling & Error Translation."""

import time
import httpx
from typing import List, Dict, Any, Optional
from app.domain.interfaces.llm import BaseLLMProvider, LLMResult
from app.domain.models.chat import ChatMessage
from app.domain.models.response import LLMResponseMetadata, TokenUsage
from app.utils.logger import logger
from app.utils.exceptions import (
    LLMAuthenticationError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMAPIError,
)


class AbstractHTTPLLMProvider(BaseLLMProvider):
    """Base class for HTTP REST API based LLM providers with automatic error handling."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key=api_key, model_name=model_name, **kwargs)
        self.timeout = timeout

    async def _safe_http_post(self, url: str, headers: Dict[str, str], json_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform an async HTTP POST request with standardized error handling and logging."""
        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers, json=json_payload)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                if response.status_code == 401 or response.status_code == 403:
                    logger.error(f"LLM Auth failure ({response.status_code}) for model {self.model_name}: {response.text}")
                    raise LLMAuthenticationError(
                        f"Authentication failed for model '{self.model_name}'. Check your API key.",
                        details={"status_code": response.status_code, "body": response.text}
                    )
                elif response.status_code == 429:
                    logger.warning(f"LLM Rate Limit / Quota exceeded for model {self.model_name}")
                    raise LLMRateLimitError(
                        f"Rate limit or quota exceeded for model '{self.model_name}'.",
                        details={"status_code": 429, "body": response.text}
                    )
                elif response.status_code != 200:
                    logger.error(f"LLM API Error ({response.status_code}): {response.text}")
                    raise LLMAPIError(
                        f"API Error from provider endpoint ({response.status_code}): {response.text}",
                        details={"status_code": response.status_code, "body": response.text}
                    )

                data = response.json()
                logger.info(f"LLM request succeeded | Provider: {self.__class__.__name__} | Model: {self.model_name} | Latency: {elapsed_ms:.1f}ms")
                return data

        except httpx.TimeoutException as exc:
            logger.error(f"LLM Timeout after {self.timeout}s calling {url}")
            raise LLMTimeoutError(f"Request to LLM provider timed out after {self.timeout} seconds.") from exc
        except httpx.NetworkError as exc:
            logger.error(f"LLM Network failure calling {url}: {exc}")
            raise LLMAPIError(f"Network failure connecting to LLM provider endpoint: {exc}") from exc
        except Exception as exc:
            if isinstance(exc, (LLMAuthenticationError, LLMRateLimitError, LLMTimeoutError, LLMAPIError)):
                raise exc
            logger.error(f"Unexpected LLM API error: {exc}")
            raise LLMAPIError(f"Unexpected error calling LLM provider: {exc}") from exc
