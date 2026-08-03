"""Ollama Local LLM Provider Adapter."""

import time
import httpx
from typing import List, Any
from app.llm.base import AbstractHTTPLLMProvider
from app.domain.interfaces.llm import LLMResult
from app.domain.models.chat import ChatMessage
from app.domain.models.response import LLMResponseMetadata, TokenUsage
from app.config.settings import settings


class OllamaProvider(AbstractHTTPLLMProvider):
    """Adapter for Ollama Local LLM API."""

    def __init__(self, api_key: str = "", model_name: str = "llama3.1", timeout: float = 120.0, **kwargs: Any) -> None:
        super().__init__(api_key=api_key, model_name=model_name, timeout=timeout, **kwargs)
        base_url = kwargs.get("base_url") or settings.OLLAMA_BASE_URL
        self.endpoint = f"{base_url.rstrip('/')}/api/chat"

    async def generate_completion(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResult:
        headers = {"Content-Type": "application/json"}

        payload = {
            "model": self.model_name,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 8192,
            },
        }

        start_time = time.perf_counter()
        raw_data = await self._safe_http_post(self.endpoint, headers, payload)
        latency_ms = (time.perf_counter() - start_time) * 1000

        content = raw_data.get("message", {}).get("content", "")
        token_usage = TokenUsage(
            prompt_tokens=raw_data.get("prompt_eval_count", 0),
            completion_tokens=raw_data.get("eval_count", 0),
            total_tokens=raw_data.get("prompt_eval_count", 0) + raw_data.get("eval_count", 0),
        )

        metadata = LLMResponseMetadata(
            provider="ollama",
            model=self.model_name,
            latency_ms=round(latency_ms, 2),
            token_usage=token_usage,
            finish_reason="stop" if raw_data.get("done") else None,
        )

        return LLMResult(content=content, metadata=metadata, raw_response=raw_data)

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(self.endpoint.replace("/api/chat", "/api/tags"))
                return res.status_code == 200
        except Exception:
            return False
