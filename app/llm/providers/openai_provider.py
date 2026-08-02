"""OpenAI LLM Provider Adapter."""

import time
from typing import List, Any
from app.llm.base import AbstractHTTPLLMProvider
from app.domain.interfaces.llm import LLMResult
from app.domain.models.chat import ChatMessage
from app.domain.models.response import LLMResponseMetadata, TokenUsage
from app.utils.exceptions import LLMAuthenticationError


class OpenAIProvider(AbstractHTTPLLMProvider):
    """Adapter for OpenAI Chat Completions API."""

    def __init__(self, api_key: str, model_name: str = "gpt-4o", timeout: float = 60.0, **kwargs: Any) -> None:
        super().__init__(api_key=api_key, model_name=model_name, timeout=timeout, **kwargs)
        self.endpoint = "https://api.openai.com/v1/chat/completions"

    async def generate_completion(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResult:
        if not self.api_key:
            raise LLMAuthenticationError("OPENAI_API_KEY is not configured.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        start_time = time.perf_counter()
        raw_data = await self._safe_http_post(self.endpoint, headers, payload)
        latency_ms = (time.perf_counter() - start_time) * 1000

        content = raw_data["choices"][0]["message"]["content"] or ""
        usage_raw = raw_data.get("usage", {})
        token_usage = TokenUsage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        )

        metadata = LLMResponseMetadata(
            provider="openai",
            model=self.model_name,
            latency_ms=round(latency_ms, 2),
            token_usage=token_usage,
            finish_reason=raw_data["choices"][0].get("finish_reason"),
        )

        return LLMResult(content=content, metadata=metadata, raw_response=raw_data)

    async def health_check(self) -> bool:
        """Verify API key validity with a lightweight test call."""
        if not self.api_key:
            return False
        try:
            test_msg = [ChatMessage(role="user", content="ping")]
            res = await self.generate_completion(test_msg, max_tokens=5)
            return bool(res.content)
        except Exception:
            return False
