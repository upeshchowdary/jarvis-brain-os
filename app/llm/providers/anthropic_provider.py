"""Anthropic Claude LLM Provider Adapter."""

import time
from typing import List, Any
from app.llm.base import AbstractHTTPLLMProvider
from app.domain.interfaces.llm import LLMResult
from app.domain.models.chat import ChatMessage
from app.domain.models.response import LLMResponseMetadata, TokenUsage
from app.utils.exceptions import LLMAuthenticationError


class AnthropicProvider(AbstractHTTPLLMProvider):
    """Adapter for Anthropic Messages API."""

    def __init__(self, api_key: str, model_name: str = "claude-3-5-sonnet-20240620", timeout: float = 60.0, **kwargs: Any) -> None:
        super().__init__(api_key=api_key, model_name=model_name, timeout=timeout, **kwargs)
        self.endpoint = "https://api.anthropic.com/v1/messages"

    async def generate_completion(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResult:
        if not self.api_key:
            raise LLMAuthenticationError("ANTHROPIC_API_KEY is not configured.")

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        # Separate system messages from user/assistant stream per Anthropic API spec
        system_prompt = None
        formatted_messages = []
        for msg in messages:
            if msg.role.value in ("system", "developer"):
                system_prompt = msg.content
            else:
                formatted_messages.append({"role": msg.role.value, "content": msg.content})

        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            payload["system"] = system_prompt

        start_time = time.perf_counter()
        raw_data = await self._safe_http_post(self.endpoint, headers, payload)
        latency_ms = (time.perf_counter() - start_time) * 1000

        content = ""
        if "content" in raw_data and len(raw_data["content"]) > 0:
            content = raw_data["content"][0].get("text", "")

        usage_raw = raw_data.get("usage", {})
        input_tokens = usage_raw.get("input_tokens", 0)
        output_tokens = usage_raw.get("output_tokens", 0)
        token_usage = TokenUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

        metadata = LLMResponseMetadata(
            provider="anthropic",
            model=self.model_name,
            latency_ms=round(latency_ms, 2),
            token_usage=token_usage,
            finish_reason=raw_data.get("stop_reason"),
        )

        return LLMResult(content=content, metadata=metadata, raw_response=raw_data)

    async def health_check(self) -> bool:
        if not self.api_key:
            return False
        try:
            test_msg = [ChatMessage(role="user", content="ping")]
            res = await self.generate_completion(test_msg, max_tokens=5)
            return bool(res.content)
        except Exception:
            return False
