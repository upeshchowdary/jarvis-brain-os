"""Model Manager for Groq API interaction, model switching, streaming, fallback, and health checks."""

import time
import httpx
from typing import List, Dict, Any, Optional, AsyncGenerator
from brain.brain_config import brain_config
from brain.logger import logger


class ModelManager:
    """Manages LLM connectivity via Groq Cloud API with model switching, fallback, and streaming support."""

    def __init__(self, api_key: Optional[str] = None, current_model: Optional[str] = None) -> None:
        self.api_key = api_key or brain_config.GROQ_API_KEY
        self.current_model = current_model or brain_config.DEFAULT_MODEL
        self.endpoint = f"{brain_config.GROQ_BASE_URL.rstrip('/')}/chat/completions"
        self.available_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "deepseek-r1-distill-llama-70b",
        ]

    def switch_model(self, new_model: str) -> bool:
        """Switch active LLM model dynamically."""
        if not new_model:
            return False
        logger.info(f"Switching ModelManager active model from '{self.current_model}' to '{new_model}'")
        self.current_model = new_model
        return True

    def list_models(self) -> List[str]:
        """Return list of supported Groq models."""
        return self.available_models.copy()

    async def health_check(self) -> bool:
        """Verify API key and Groq endpoint availability."""
        if not self.api_key:
            logger.warning("Groq API Key is not set in environment or configuration.")
            return False
        try:
            test_messages = [{"role": "user", "content": "ping"}]
            res = await self.generate(messages=test_messages, max_tokens=5)
            return bool(res.get("content"))
        except Exception as exc:
            logger.error(f"ModelManager health check failed: {exc}")
            return False

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute async completion request against Groq API with automatic fallback on rate limit (429)."""
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not configured in .env file.")

        models_to_try = [
            model or self.current_model,
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ]
        # Remove duplicates preserving order
        unique_models = []
        for m in models_to_try:
            if m and m not in unique_models:
                unique_models.append(m)

        temp = temperature if temperature is not None else brain_config.TEMPERATURE
        tokens = max_tokens or brain_config.MAX_TOKENS
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_exception = None

        for target_model in unique_models:
            payload = {
                "model": target_model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": tokens,
            }

            start_time = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=brain_config.TIMEOUT_SECONDS) as client:
                    response = await client.post(self.endpoint, headers=headers, json=payload)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000

                    if response.status_code == 429:
                        logger.warning(
                            f"Model '{target_model}' hit Groq rate limit (429 TPD limit). "
                            f"Attempting automatic fallback to next model in queue..."
                        )
                        last_exception = RuntimeError(f"Rate limit exceeded for model {target_model}")
                        continue

                    if response.status_code != 200:
                        logger.error(f"Groq API error ({response.status_code}) for model {target_model}: {response.text}")
                        last_exception = RuntimeError(f"Groq API error [{response.status_code}]: {response.text}")
                        continue

                    data = response.json()
                    choice = data["choices"][0]
                    content = choice["message"]["content"] or ""
                    usage = data.get("usage", {})

                    logger.info(
                        f"ModelManager generation succeeded | Model: {target_model} | "
                        f"Tokens: {usage.get('total_tokens', 0)} | Latency: {elapsed_ms:.1f}ms"
                    )

                    return {
                        "content": content,
                        "model": target_model,
                        "provider": "groq",
                        "latency_ms": round(elapsed_ms, 2),
                        "finish_reason": choice.get("finish_reason"),
                        "usage": {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        },
                    }

            except Exception as exc:
                logger.error(f"Exception calling Groq API for model {target_model}: {exc}")
                last_exception = exc
                continue

        # If all models failed or hit rate limit
        logger.error(f"All Groq models failed. Last exception: {last_exception}")
        return {
            "content": "I apologize, but all LLM models currently reached their daily free quota limit on Groq API. Please wait a short while or provide an additional API key.",
            "model": "fallback_exhausted",
            "provider": "groq",
            "latency_ms": 0.0,
            "finish_reason": "rate_limit_fallback_exhausted",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream completion tokens asynchronously from Groq API."""
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not configured.")

        target_model = model or self.current_model
        temp = temperature if temperature is not None else brain_config.TEMPERATURE
        tokens = max_tokens or brain_config.MAX_TOKENS

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=brain_config.TIMEOUT_SECONDS) as client:
            async with client.stream("POST", self.endpoint, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    raise RuntimeError(f"Groq API stream failed [{response.status_code}]: {error_body.decode('utf-8')}")

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = httpx.json.loads(data_str)
                            delta = chunk["choices"][0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                yield token
                        except Exception:
                            continue


# Global model manager singleton
model_manager = ModelManager()
